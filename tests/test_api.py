from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    """测试健康检查接口"""
    response = client.get("/health")
    assert response.status_code == 200

def test_run_rejects_empty_goal():
    """测试任务目标为空时返回 422"""
    response = client.post(
        "/api/agent/run",
        json={"goal":""}
    )

    assert response.status_code == 422

def test_run_rejects_missing_goal():
    """测试缺少任务目标时返回 422"""
    response = client.post(
        "/api/agent/run",
        json={}
    )

    assert response.status_code == 422

def test_confirm_rejects_empty_goal():
    """测试会话 ID 为空时返回 422"""
    response = client.post(
        "/api/agent/confirm",
        json={
            "thread_id":"",
            "approved":False,
        }
    )
    assert response.status_code == 422