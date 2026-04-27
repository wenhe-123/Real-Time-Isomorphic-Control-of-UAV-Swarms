from pyorbbecsdk import *

ctx = Context()
dev_list = ctx.query_devices()
if dev_list.get_count() == 0:
    raise RuntimeError("No device found")

device = dev_list.get_device_by_index(0)
pipeline = Pipeline(device)

for sensor_type, name in [
    (OBSensorType.COLOR_SENSOR, "COLOR"),
    (OBSensorType.DEPTH_SENSOR, "DEPTH"),
]:
    print(f"\n{name} profiles:")
    profiles = pipeline.get_stream_profile_list(sensor_type)
    count = profiles.get_count()
    print("count =", count)
    for i in range(count):
        p = profiles.get_video_stream_profile_by_index(i)
        print(
            i,
            "width=", p.get_width(),
            "height=", p.get_height(),
            "fps=", p.get_fps(),
            "format=", p.get_format(),
        )
