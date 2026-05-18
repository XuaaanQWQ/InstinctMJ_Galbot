"""Load and visualize scene_mjx.xml from current directory."""

import mujoco
import mujoco.viewer


def main():
    # 加载 XML 文件
    model = mujoco.MjModel.from_xml_path("scene_mjx.xml")

    # 设置重力为 0（零重力环境）
    model.opt.gravity = [0, 0, 0]

    # 创建数据对象
    data = mujoco.MjData(model)

    print(f"Model loaded successfully!")
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of actuators: {model.nu}")
    print(f"Number of sensors: {model.nsensordata}")

    # 启动 MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nViewer opened. Press Ctrl+C to exit.")

        # 运行仿真循环
        while viewer.is_running():
            # 执行一步仿真
            mujoco.mj_step(model, data)

            # 同步 viewer
            viewer.sync()

            # 控制仿真速度（可选）
            # import time
            # time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
