import unittest

from filtering.car_size_filter import (
    LargeCarFilterConfig,
    apply_large_car_to_truck,
)


def frame(frame_id, detections):
    return {
        "frame_id": str(frame_id),
        "num_detections": len(detections),
        "detections": detections,
    }


def detection(class_name, track_id, length, width=2.0):
    return {
        "class_name": class_name,
        "track_id": track_id,
        "score": 0.8,
        "box_lidar": [0.0, 0.0, 0.0, length, width, 1.7, 0.0],
    }


class LargeCarFilterTest(unittest.TestCase):
    def test_relabels_large_car_track_and_leaves_normal_classes_untouched(self):
        source = [
            frame(0, [
                detection("Car", 1, 6.3),
                detection("Car", 2, 4.6),
                detection("Truck", 3, 8.0),
            ]),
            frame(1, [
                detection("Car", 1, 6.1),
                detection("Car", 2, 4.7),
            ]),
        ]

        output, stats = apply_large_car_to_truck(source)

        self.assertEqual(
            [d["class_name"] for d in output[0]["detections"]],
            ["Truck", "Car", "Truck"],
        )
        self.assertEqual(output[1]["detections"][0]["class_name"], "Truck")
        self.assertEqual(stats["large_car_tracks_relabelled"], 1)
        self.assertEqual(stats["large_car_detections_relabelled"], 2)
        self.assertEqual(source[0]["detections"][0]["class_name"], "Car")

    def test_track_median_controls_threshold(self):
        source = [
            frame(0, [detection("Car", 1, 6.4)]),
            frame(1, [detection("Car", 1, 5.8)]),
            frame(2, [detection("Car", 1, 6.2)]),
        ]

        output, stats = apply_large_car_to_truck(
            source, LargeCarFilterConfig(truck_length_min=6.0))

        self.assertTrue(all(
            d["class_name"] == "Truck"
            for f in output for d in f["detections"]))
        self.assertEqual(stats["large_car_detections_relabelled"], 3)


if __name__ == "__main__":
    unittest.main()
