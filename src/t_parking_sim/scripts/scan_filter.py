#!/usr/bin/env python3
"""Remove moving objects from a LaserScan and republish the static remainder.

The node clusters each scan, transforms every cluster centroid into the odom
frame, and differences consecutive centroids *in that frame* to obtain a
velocity.  Because odom is a fixed frame, the vehicle's own translation and
rotation cancel out of that difference automatically -- there is deliberately
no separate ego-motion compensation term anywhere in this file.  An earlier
attempt measured velocities in the vehicle frame and corrected them with a
scalar `vx += ego_speed`; that ignores the yaw rate entirely, and during a
parking manoeuvre (large steering angles) it made stationary walls look like
fast movers, inverting the classification.

Classification is deliberately conservative: only tracks confirmed dynamic
over several frames are erased.  Brand-new tracks (whose velocity estimate is
still noisy) and tracks sitting between the two speed thresholds keep their
previous label, so a static obstacle is never dropped on a single bad frame.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan

import tf2_ros

UNKNOWN = 'unknown'
STATIC = 'static'
DYNAMIC = 'dynamic'


@dataclass
class Cluster:
    """One contiguous run of scan returns, expressed in the odom frame."""

    cx: float
    cy: float
    extent: float
    truncated: bool
    indices: np.ndarray = field(repr=False)


@dataclass
class Track:
    """A cluster followed across scans.  All positions are odom frame."""

    track_id: int
    cx: float
    cy: float
    stamp: float
    extent: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    # Reference position for the net-displacement gate; see _update_tracks.
    ref_cx: float = 0.0
    ref_cy: float = 0.0
    ref_stamp: float = 0.0
    net_moved: bool = False
    age: int = 1
    missed: int = 0
    dynamic_count: int = 0
    static_count: int = 0
    label: str = UNKNOWN

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


class ScanFilter(Node):

    def __init__(self) -> None:
        super().__init__('scan_filter')

        self.declare_parameter('input_scan_topic', '/scan')
        self.declare_parameter('static_scan_topic', '/scan_static')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('tf_timeout_sec', 0.05)
        self.declare_parameter('max_scan_delay_sec', 0.5)
        self.declare_parameter('min_range', 0.15)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('cluster_distance_threshold', 0.25)
        self.declare_parameter('cluster_distance_scale', 0.04)
        self.declare_parameter('min_cluster_points', 3)
        self.declare_parameter('max_cluster_width', 3.0)
        self.declare_parameter('track_match_distance', 0.8)
        self.declare_parameter('max_missed_frames', 4)
        self.declare_parameter('velocity_filter_alpha', 0.6)
        self.declare_parameter('dynamic_speed_threshold', 0.30)
        self.declare_parameter('static_speed_threshold', 0.15)
        self.declare_parameter('dynamic_confirm_frames', 3)
        self.declare_parameter('static_confirm_frames', 2)
        self.declare_parameter('min_track_age_for_dynamic', 3)
        self.declare_parameter('max_extent_change', 0.25)
        self.declare_parameter('displacement_window_sec', 0.6)
        self.declare_parameter('min_net_displacement', 0.20)
        self.declare_parameter('dynamic_dilation_rad', 0.02)
        self.declare_parameter('debug_enable', False)

        self.input_scan_topic = str(
            self.get_parameter('input_scan_topic').value)
        self.static_scan_topic = str(
            self.get_parameter('static_scan_topic').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.tf_timeout_sec = float(self.get_parameter('tf_timeout_sec').value)
        self.max_scan_delay_sec = float(
            self.get_parameter('max_scan_delay_sec').value)
        self.min_range = float(self.get_parameter('min_range').value)
        self.max_range = float(self.get_parameter('max_range').value)
        self.cluster_distance_threshold = float(
            self.get_parameter('cluster_distance_threshold').value)
        self.cluster_distance_scale = float(
            self.get_parameter('cluster_distance_scale').value)
        self.min_cluster_points = int(
            self.get_parameter('min_cluster_points').value)
        self.max_cluster_width = float(
            self.get_parameter('max_cluster_width').value)
        self.track_match_distance = float(
            self.get_parameter('track_match_distance').value)
        self.max_missed_frames = int(
            self.get_parameter('max_missed_frames').value)
        self.velocity_filter_alpha = float(
            self.get_parameter('velocity_filter_alpha').value)
        self.dynamic_speed_threshold = float(
            self.get_parameter('dynamic_speed_threshold').value)
        self.static_speed_threshold = float(
            self.get_parameter('static_speed_threshold').value)
        self.dynamic_confirm_frames = int(
            self.get_parameter('dynamic_confirm_frames').value)
        self.static_confirm_frames = int(
            self.get_parameter('static_confirm_frames').value)
        self.min_track_age_for_dynamic = int(
            self.get_parameter('min_track_age_for_dynamic').value)
        self.max_extent_change = float(
            self.get_parameter('max_extent_change').value)
        self.displacement_window_sec = float(
            self.get_parameter('displacement_window_sec').value)
        self.min_net_displacement = float(
            self.get_parameter('min_net_displacement').value)
        self.dynamic_dilation_rad = float(
            self.get_parameter('dynamic_dilation_rad').value)
        self.debug_enable = bool(self.get_parameter('debug_enable').value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0
        self.angles: Optional[np.ndarray] = None
        self.angles_signature: Optional[Tuple[int, float, float]] = None
        self.tf_failure_logged = False
        # Scans arrive ~0.15 s before the odom transform covering their stamp,
        # so a scan is held here until its transform lands rather than blocked
        # on inside the 15 Hz callback.  See _drain_pending.
        self.pending: Deque[LaserScan] = deque()
        self.latest_scan_stamp = 0.0

        self.publisher = self.create_publisher(
            LaserScan, self.static_scan_topic, qos_profile_sensor_data)
        self.subscription = self.create_subscription(
            LaserScan, self.input_scan_topic, self._scan_callback,
            qos_profile_sensor_data)

        # The marker publisher must not exist at all when debugging is off:
        # an always-created publisher would still advertise the topic and show
        # up in rqt_graph, which is exactly what this package is trying to
        # keep clean.
        self.debug_publisher = None
        if self.debug_enable:
            from visualization_msgs.msg import Marker, MarkerArray
            self._marker_cls = Marker
            self._marker_array_cls = MarkerArray
            self.debug_publisher = self.create_publisher(
                MarkerArray, '~/debug_markers', 1)

        self.get_logger().info(
            f'scan_filter: {self.input_scan_topic} -> '
            f'{self.static_scan_topic}, velocity measured in '
            f"'{self.odom_frame}', debug={self.debug_enable}")

    # ------------------------------------------------------------------
    # [1] point expansion
    # ------------------------------------------------------------------

    def _scan_angles(self, scan: LaserScan) -> np.ndarray:
        """Return the per-index bearing array, cached across scans."""
        signature = (len(scan.ranges), scan.angle_min, scan.angle_increment)
        if self.angles_signature != signature or self.angles is None:
            self.angles = (
                scan.angle_min
                + np.arange(len(scan.ranges), dtype=np.float64)
                * scan.angle_increment)
            self.angles_signature = signature
        return self.angles

    def _lookup_yaw_translation(
            self, scan: LaserScan,
            timeout_sec: float) -> Optional[Tuple[float, float, float]]:
        """One TF lookup per scan -- never one per point."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.odom_frame,
                scan.header.frame_id,
                Time.from_msg(scan.header.stamp),
                timeout=Duration(seconds=timeout_sec))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException, tf2_ros.TransformException):
            return None
        translation = transform.transform.translation
        q = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return translation.x, translation.y, yaw

    # ------------------------------------------------------------------
    # [2] clustering
    # ------------------------------------------------------------------

    def _cluster(
            self,
            ox: np.ndarray,
            oy: np.ndarray,
            ranges: np.ndarray,
            indices: np.ndarray) -> List[Cluster]:
        if indices.size == 0:
            return []
        if indices.size == 1:
            boundaries = np.array([0, 1])
        else:
            gaps = np.hypot(np.diff(ox), np.diff(oy))
            limits = (self.cluster_distance_threshold
                      + self.cluster_distance_scale * ranges[1:])
            breaks = np.flatnonzero(gaps > limits) + 1
            boundaries = np.concatenate(
                ([0], breaks, [indices.size])).astype(int)

        clusters: List[Cluster] = []
        last = indices.size
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end - start < self.min_cluster_points:
                continue
            cx_slice = ox[start:end]
            cy_slice = oy[start:end]
            width = math.hypot(
                float(cx_slice.max() - cx_slice.min()),
                float(cy_slice.max() - cy_slice.min()))
            if width > self.max_cluster_width:
                continue
            clusters.append(Cluster(
                cx=float(cx_slice.mean()),
                cy=float(cy_slice.mean()),
                extent=width,
                # A cluster running into the first or last return is cut off
                # by the field of view rather than by its own edge, so how
                # much of it we see -- and therefore where its centroid sits
                # -- changes as the vehicle yaws.  See _update_tracks.
                truncated=(start == 0 or end == last),
                indices=indices[start:end]))
        return clusters

    # ------------------------------------------------------------------
    # [3] tracking
    # ------------------------------------------------------------------

    def _match(
            self,
            track_ids: List[int],
            clusters: List[Cluster]) -> Dict[int, int]:
        """Greedy nearest-first 1:1 assignment of tracks to clusters."""
        candidates = []
        for track_id in track_ids:
            track = self.tracks[track_id]
            for cluster_index, cluster in enumerate(clusters):
                distance = math.hypot(
                    cluster.cx - track.cx, cluster.cy - track.cy)
                if distance <= self.track_match_distance:
                    candidates.append((distance, track_id, cluster_index))
        candidates.sort(key=lambda item: item[0])

        assignment: Dict[int, int] = {}
        used_tracks = set()
        used_clusters = set()
        for _, track_id, cluster_index in candidates:
            if track_id in used_tracks or cluster_index in used_clusters:
                continue
            used_tracks.add(track_id)
            used_clusters.add(cluster_index)
            assignment[track_id] = cluster_index
        return assignment

    def _update_tracks(
            self,
            clusters: List[Cluster],
            stamp: float) -> Dict[int, int]:
        assignment = self._match(list(self.tracks.keys()), clusters)
        alpha = self.velocity_filter_alpha

        for track_id, cluster_index in assignment.items():
            track = self.tracks[track_id]
            cluster = clusters[cluster_index]

            # Differencing centroids only measures motion when both frames saw
            # the same part of the object.  When an occluder slides across a
            # wall, or the field-of-view edge sweeps over it as the vehicle
            # yaws, the visible fragment grows or shrinks and its centroid
            # slides with it -- at several m/s during a hard steering input --
            # while the wall itself never moved.  Treating that as velocity is
            # how a static obstacle gets erased, so an unstable observation
            # yields no dynamic evidence at all.
            stable = (not cluster.truncated
                      and abs(cluster.extent - track.extent)
                      <= self.max_extent_change)

            dt = stamp - track.stamp
            if stable and dt > 0.0:
                # Both centroids are odom-frame, so this difference already
                # excludes the vehicle's own motion.  Do not add any further
                # ego-motion correction here.
                vx_measured = (cluster.cx - track.cx) / dt
                vy_measured = (cluster.cy - track.cy) / dt
                track.vx = alpha * vx_measured + (1.0 - alpha) * track.vx
                track.vy = alpha * vy_measured + (1.0 - alpha) * track.vy
                track.stamp = stamp
            elif not stable:
                track.stamp = stamp
            track.cx = cluster.cx
            track.cy = cluster.cy
            track.extent = cluster.extent
            track.age += 1
            track.missed = 0

            # A rigid object's centroid is not a fixed point on the object:
            # as the viewing aspect rotates, which faces are visible changes,
            # and the centroid slides by up to half the object's size.  On a
            # 0.6 m obstacle that is a plausible-looking 0.3-0.5 m/s for a few
            # frames, which alone would clear the speed threshold.  Real
            # travel differs from that wander in that it accumulates, so
            # dynamic evidence additionally requires the centroid to have
            # actually left where it was a window ago.
            if stamp - track.ref_stamp >= self.displacement_window_sec:
                net = math.hypot(
                    track.cx - track.ref_cx, track.cy - track.ref_cy)
                track.net_moved = net >= self.min_net_displacement
                track.ref_cx = track.cx
                track.ref_cy = track.cy
                track.ref_stamp = stamp

            self._classify(track, stable)

        matched_clusters = set(assignment.values())
        for cluster_index, cluster in enumerate(clusters):
            if cluster_index in matched_clusters:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = Track(
                track_id=track_id, cx=cluster.cx, cy=cluster.cy, stamp=stamp,
                extent=cluster.extent, ref_cx=cluster.cx, ref_cy=cluster.cy,
                ref_stamp=stamp)
            assignment[track_id] = cluster_index

        for track_id in list(self.tracks.keys()):
            if track_id in assignment:
                continue
            track = self.tracks[track_id]
            track.missed += 1
            if track.missed > self.max_missed_frames:
                del self.tracks[track_id]
        return assignment

    # ------------------------------------------------------------------
    # [4] classification
    # ------------------------------------------------------------------

    def _classify(self, track: Track, stable: bool = True) -> None:
        speed = track.speed
        old_enough = track.age >= self.min_track_age_for_dynamic

        if not stable:
            # No usable velocity this frame.  Decay the dynamic evidence and
            # keep whatever label the track already had.
            track.dynamic_count = max(track.dynamic_count - 1, 0)
            if track.dynamic_count < self.dynamic_confirm_frames:
                if track.label == DYNAMIC:
                    track.label = UNKNOWN
            return

        if old_enough and track.net_moved and (
                speed >= self.dynamic_speed_threshold):
            track.dynamic_count = min(
                track.dynamic_count + 1, self.dynamic_confirm_frames)
            track.static_count = 0
        else:
            track.dynamic_count = max(track.dynamic_count - 1, 0)

        if speed <= self.static_speed_threshold:
            track.static_count = min(
                track.static_count + 1, self.static_confirm_frames)

        if old_enough and track.dynamic_count >= self.dynamic_confirm_frames:
            track.label = DYNAMIC
        elif track.static_count >= self.static_confirm_frames:
            track.label = STATIC
        # Otherwise the previous label is kept: between the two speed
        # thresholds the evidence is inconclusive, and a freshly created
        # track stays UNKNOWN, which is never erased.

    # ------------------------------------------------------------------
    # [5] output scan
    # ------------------------------------------------------------------

    def _build_static_scan(
            self,
            scan: LaserScan,
            removed: np.ndarray) -> LaserScan:
        out = LaserScan()
        # Every geometric field must survive untouched; slam_toolbox rejects a
        # scan whose frame or angular layout does not match what it expects.
        out.header = scan.header
        out.angle_min = scan.angle_min
        out.angle_max = scan.angle_max
        out.angle_increment = scan.angle_increment
        out.time_increment = scan.time_increment
        out.scan_time = scan.scan_time
        out.range_min = scan.range_min
        out.range_max = scan.range_max
        out.intensities = scan.intensities

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if removed.size:
            ranges = ranges.copy()
            ranges[removed] = float('inf')
        out.ranges = ranges.tolist()
        return out

    def _removed_indices(
            self,
            scan: LaserScan,
            clusters: List[Cluster],
            assignment: Dict[int, int]) -> np.ndarray:
        dynamic_indices = [
            clusters[cluster_index].indices
            for track_id, cluster_index in assignment.items()
            if self.tracks[track_id].label == DYNAMIC
        ]
        if not dynamic_indices:
            return np.empty(0, dtype=int)

        indices = np.concatenate(dynamic_indices)
        pad = 0
        if scan.angle_increment != 0.0:
            pad = int(round(
                self.dynamic_dilation_rad / abs(scan.angle_increment)))
        if pad > 0:
            offsets = np.arange(-pad, pad + 1)
            indices = (indices[:, None] + offsets[None, :]).ravel()
        count = len(scan.ranges)
        indices = indices[(indices >= 0) & (indices < count)]
        return np.unique(indices)

    # ------------------------------------------------------------------
    # [6] debug
    # ------------------------------------------------------------------

    def _publish_debug(
            self,
            scan: LaserScan,
            clusters: List[Cluster],
            assignment: Dict[int, int]) -> None:
        if self.debug_publisher is None:
            return
        Marker = self._marker_cls
        array = self._marker_array_cls()

        clear = Marker()
        clear.header.frame_id = self.odom_frame
        clear.header.stamp = scan.header.stamp
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        colors = {
            DYNAMIC: (1.0, 0.15, 0.15),
            STATIC: (0.15, 0.8, 0.3),
            UNKNOWN: (0.6, 0.6, 0.6),
        }
        for marker_index, track_id in enumerate(sorted(assignment)):
            track = self.tracks[track_id]
            marker = Marker()
            marker.header.frame_id = self.odom_frame
            marker.header.stamp = scan.header.stamp
            marker.ns = 'scan_filter_tracks'
            marker.id = marker_index
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = track.cx
            marker.pose.position.y = track.cy
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.18
            marker.scale.y = 0.18
            marker.scale.z = 0.18
            red, green, blue = colors[track.label]
            marker.color.r = red
            marker.color.g = green
            marker.color.b = blue
            marker.color.a = 0.85
            marker.lifetime = Duration(seconds=0.5).to_msg()
            array.markers.append(marker)

            text = Marker()
            text.header = marker.header
            text.ns = 'scan_filter_labels'
            text.id = marker_index
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = track.cx
            text.pose.position.y = track.cy
            text.pose.position.z = 0.3
            text.pose.orientation.w = 1.0
            text.scale.z = 0.16
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 0.9
            text.text = (f'{track.track_id} {track.label} '
                         f'{track.speed:.2f} a{track.age}')
            text.lifetime = Duration(seconds=0.5).to_msg()
            array.markers.append(text)

        self.debug_publisher.publish(array)

    # ------------------------------------------------------------------

    def _scan_callback(self, scan: LaserScan) -> None:
        self.latest_scan_stamp = max(
            self.latest_scan_stamp,
            Time.from_msg(scan.header.stamp).nanoseconds * 1e-9)
        self.pending.append(scan)
        self._drain_pending()

    def _drain_pending(self) -> None:
        """Publish held scans in order as their odom transform becomes known.

        The bridge stamps a scan roughly 0.15 s before the odom transform
        covering that stamp reaches the buffer, so a blocking lookup inside
        the callback fails almost every time while still burning the whole
        timeout.  Holding a few scans instead costs the same 0.15 s of
        latency but lets nearly every scan be filtered.  Age is measured
        against the newest scan stamp rather than a clock, so the node
        behaves identically with or without use_sim_time.
        """
        while self.pending:
            scan = self.pending[0]
            transform = self._lookup_yaw_translation(scan, 0.0)
            if transform is not None:
                self.pending.popleft()
                self.tf_failure_logged = False
                self._process(scan, transform)
                continue

            stamp = Time.from_msg(scan.header.stamp).nanoseconds * 1e-9
            if self.latest_scan_stamp - stamp < self.max_scan_delay_sec:
                return

            # Give the transform one last short chance, then give up on it.
            # Passing the scan through unfiltered is the only safe failure
            # mode: dropping it would stall slam_toolbox and the costmaps.
            # Track state is left untouched so the next successful lookup
            # resumes from a consistent history rather than a stale one.
            transform = self._lookup_yaw_translation(scan, self.tf_timeout_sec)
            self.pending.popleft()
            if transform is not None:
                self.tf_failure_logged = False
                self._process(scan, transform)
                continue
            if not self.tf_failure_logged:
                self.get_logger().warn(
                    f'TF {self.odom_frame} <- {scan.header.frame_id} '
                    'unavailable; passing the scan through unfiltered')
                self.tf_failure_logged = True
            self.publisher.publish(scan)

    def _process(
            self,
            scan: LaserScan,
            transform: Tuple[float, float, float]) -> None:
        tx, ty, yaw = transform

        ranges = np.asarray(scan.ranges, dtype=np.float64)
        angles = self._scan_angles(scan)
        low = max(self.min_range, scan.range_min)
        high = min(self.max_range, scan.range_max)
        valid = (np.isfinite(ranges) & (ranges > 0.0)
                 & (ranges >= low) & (ranges <= high))
        indices = np.flatnonzero(valid)

        clusters: List[Cluster] = []
        assignment: Dict[int, int] = {}
        if indices.size:
            r = ranges[indices]
            a = angles[indices]
            sensor_x = r * np.cos(a)
            sensor_y = r * np.sin(a)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            odom_x = tx + cos_yaw * sensor_x - sin_yaw * sensor_y
            odom_y = ty + sin_yaw * sensor_x + cos_yaw * sensor_y

            clusters = self._cluster(odom_x, odom_y, r, indices)
            stamp = Time.from_msg(scan.header.stamp).nanoseconds * 1e-9
            assignment = self._update_tracks(clusters, stamp)

        removed = self._removed_indices(scan, clusters, assignment)
        self.publisher.publish(self._build_static_scan(scan, removed))
        self._publish_debug(scan, clusters, assignment)


def main() -> None:
    rclpy.init()
    node = ScanFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
