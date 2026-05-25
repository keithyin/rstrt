

```
trtexec   --onnx=model.onnx   --saveEngine=model.plan   --fp16   --workspace=4096   --minShapes=feature:128x200x61,length:128   --optShapes=feature:128x200x61,length:128   --maxShapes=feature:128x200x61,length:128
```