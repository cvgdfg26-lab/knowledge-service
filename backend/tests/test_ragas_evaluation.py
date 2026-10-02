import csv
import importlib.util
import json
from pathlib import Path

import pytest

from knowledge_service.rag import answer_graph


@pytest.fixture
def evaluation_program():
    """加载评估程序的轻量输入输出步骤，不导入或调用外部判分模型。"""
    path = Path(__file__).resolve().parents[2] / "eval/eval.py"
    spec = importlib.util.spec_from_file_location("evaluation_program", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dataset_requires_bom_and_reference(evaluation_program, tmp_path):
    """非 BOM 输入及空参考答案必须阻止评估，不能产生误导分数。"""
    path = tmp_path / "qa.csv"
    path.write_text("question,ground_truth\n问题,参考\n", encoding="utf-8")
    with pytest.raises(ValueError, match="BOM"):
        evaluation_program.step_2_load_test_dataset(path)
    path.write_text("question,ground_truth\n问题,\n", encoding="utf-8-sig")
    with pytest.raises(ValueError, match="为空"):
        evaluation_program.step_2_load_test_dataset(path)


def test_result_preserves_multiline_context_order_and_nan(evaluation_program, tmp_path):
    """CSV 必须保留证据顺序、引号/换行和缺失分数，不把 NaN 伪造成零分。"""
    path = tmp_path / "result.csv"
    row = {
        "question": "问题",
        "ground_truth": "参考",
        "answer": "回答。[S1]",
        "contexts": ['第一段\n带有"引号"', "第二段"],
    }
    row.update({name: 0.5 for name in evaluation_program.METRIC_COLUMNS})
    row["Faithfulness"] = float("nan")
    evaluation_program.step_6_write_result_csv([row], path)
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == evaluation_program.OUTPUT_COLUMNS
        result = next(reader)
    assert json.loads(result["contexts"]) == row["contexts"]
    assert result["Faithfulness"] == "NaN" and result["Answer Correctness"] == "0.500000"


@pytest.mark.asyncio
async def test_graph_exposes_actual_reranked_references(published):
    """评估证据必须来自图的真实重排结果，持久状态只存引用而不缓存正文。"""
    runtime = published
    expected = []

    async def rerank(question, candidates):
        """模拟不同于召回顺序的重排结果，捕获预期引用顺序。"""
        result = list(reversed(candidates))
        expected.extend(c["_id"] for c in result)
        return result

    runtime.models.rerank = rerank
    chat = runtime.domain.create_conversation("reader", "eval-chat", ["kb"])
    message = runtime.domain.reserve_answer("reader", chat["_id"], "eval-answer", "退款？")
    state = await answer_graph(runtime).ainvoke(
        {"user_id": "reader", "message_id": message["_id"]},
        {"configurable": {"thread_id": message["thread_id"]}},
    )
    assert expected and state["reranked_docs"] == expected[:5] == state["selected_ids"]
    assert all(isinstance(doc, str) for doc in state["reranked_docs"])
