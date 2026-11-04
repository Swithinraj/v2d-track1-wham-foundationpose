def _selected_metric_values(metrics, index):
    return {
        'observed_visible_pixels': int(metrics.observed_visible_pixels),
        'rendered_visible_pixels': int(metrics.rendered_visible_pixels[index]),
        'iou': float(metrics.iou[index]),
        'centroid_distance': float(metrics.centroid_distance[index]),
    }
