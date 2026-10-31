"""Initializer defaults remain live graph inputs when callers override them."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper


def _model(kind):
    if kind == "MatMul":
        x = np.arange(6,dtype=np.float32).reshape(2,3)/10
        weight = np.arange(12,dtype=np.float32).reshape(3,4)/10
        dtype = TensorProto.FLOAT
        shape = [2,4]
    elif kind == "Add":
        x = np.arange(6,dtype=np.float32).reshape(2,3)
        weight = np.array(0.5,np.float32)
        dtype = TensorProto.FLOAT
        shape = [2,3]
    else:
        x = np.arange(15,dtype=np.float32).reshape(5,3)
        weight = np.array([0,2],np.int64)
        dtype = TensorProto.INT64
        shape = [2,3]
    info = helper.make_tensor_value_info
    graph = helper.make_graph(
        [helper.make_node(kind,["x","weight"],["y"])],"overridable",
        [info("x",TensorProto.FLOAT,list(x.shape)),info("weight",dtype,list(weight.shape))],
        [info("y",TensorProto.FLOAT,shape)], [numpy_helper.from_array(weight,"weight")])
    model = helper.make_model(graph,opset_imports=[helper.make_opsetid("",18)],ir_version=10)
    onnx.checker.check_model(model,full_check=True)
    return model.SerializeToString(),x,weight


def _session(blob,provider):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    if provider == "MLXExecutionProvider":
        options.add_session_config_entry("session.disable_cpu_ep_fallback","1")
    return ort.InferenceSession(blob,options,providers=[provider])


@pytest.mark.parametrize("kind",["MatMul","Add","Gather"])
def test_default_override_and_restore(kind):
    blob,x,default = _model(kind)
    mlx = _session(blob,"MLXExecutionProvider")
    cpu = _session(blob,"CPUExecutionProvider")
    original = default.tobytes()
    for step in [None,1,2,None,3,None]:
        feeds = {"x":x}
        weight = default
        if step is not None:
            weight = ((default+step)%5 if kind == "Gather" else default+np.float32(step))
            weight = np.asarray(weight)
            feeds["weight"] = weight
        expected = x@weight if kind == "MatMul" else x+weight if kind == "Add" else x[weight]
        np.testing.assert_allclose(mlx.run(None,feeds)[0],expected,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(mlx.run(None,feeds)[0],cpu.run(None,feeds)[0],rtol=1e-6,atol=1e-6)
        assert default.tobytes() == original


def test_iobinding_reads_updated_override():
    blob,x,default = _model("MatMul")
    runtime = _session(blob,"MLXExecutionProvider")
    weight = default.copy()
    value = ort.OrtValue.ortvalue_from_numpy(weight)
    binding = runtime.io_binding()
    binding.bind_cpu_input("x",x)
    binding.bind_ortvalue_input("weight",value)
    binding.bind_output("y","cpu")
    for step in range(4):
        weight[:] = default+np.float32(step)
        runtime.run_with_iobinding(binding)
        np.testing.assert_allclose(binding.copy_outputs_to_cpu()[0],x@weight,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(runtime.run(None,{"x":x})[0],x@default,rtol=1e-6,atol=1e-6)
