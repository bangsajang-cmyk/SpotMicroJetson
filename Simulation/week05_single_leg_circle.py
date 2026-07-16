import os
import time

import pybullet as p


def main():
    # 현재 파일이 있는 Simulation 폴더를 기준으로 경로 설정
    simulation_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(simulation_dir)

    client_id = p.connect(p.GUI)

    if client_id < 0:
        raise RuntimeError("PyBullet GUI 연결에 실패했습니다.")

    p.setGravity(0, 0, -9.81)
    # URDF 파일 없이 바닥 직접 생성
    plane_collision = p.createCollisionShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[5.0, 5.0, 0.01],
    )

    plane_visual = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[5.0, 5.0, 0.01],
    rgbaColor=[0.75, 0.75, 0.75, 1.0],
    )

    p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=plane_collision,
    baseVisualShapeIndex=plane_visual,
    basePosition=[0, 0, -0.01],
    )

    # URDF 파일 없이 바닥 직접 생성
    plane_collision = p.createCollisionShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[5.0, 5.0, 0.01],
    )

    plane_visual = p.createVisualShape(
    shapeType=p.GEOM_BOX,
    halfExtents=[5.0, 5.0, 0.01],
    rgbaColor=[0.75, 0.75, 0.75, 1.0],
    )

    p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=plane_collision,
    baseVisualShapeIndex=plane_visual,
    basePosition=[0, 0, -0.01],
    )

    # SpotMicro 로봇
    robot_path = os.path.join(
        simulation_dir,
        "..",
        "urdf",
        "spotmicroai_gen.urdf.xml",
    )

    if not os.path.exists(robot_path):
        raise FileNotFoundError(
            f"로봇 URDF 파일을 찾을 수 없습니다: {robot_path}"
        )

    robot_id = p.loadURDF(
        robot_path,
        basePosition=[0, 0, 0.3],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION,
    )

    print(f"SpotMicro 로딩 성공, robot_id={robot_id}")

    # 로봇이 잘 보이도록 카메라 설정
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=50,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.25],
    )

    while p.isConnected():
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    main()