import torch, vllm._custom_ops as ops
p=torch.cuda.get_device_properties(0); print(p.name, p.gcnArchName, "MPC=",p.multi_processor_count)
for (M,K) in [(320,10240),(10240,320),(512,2560),(48,2560)]:
  W=torch.randn(M,K,device="cuda:0",dtype=torch.bfloat16)
  for n in (1,5):
    X=torch.randn(n,K,device="cuda:0",dtype=torch.bfloat16)
    ref=torch.nn.functional.linear(X.float(),W.float())
    for cu in (32,64,96):
      try:
        o=ops.wvSplitK(W,X,cu,None); e=(o.float()-ref).abs().max().item()
        print(f"M{M} K{K} n{n} cu{cu} shape{tuple(o.shape)} maxabs{e:.4g}")
      except Exception as ex: print(f"M{M} K{K} n{n} cu{cu} ERR {ex}")
for m in ("aiter.ops.triton.gemm_a16w16","aiter.tuned_gemm"):
  try:
    __import__(m); print("import ok",m)
  except Exception as ex: print("import FAIL",m,repr(ex)[:150])
