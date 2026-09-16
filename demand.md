
这是一个 tensorrt 的 rust wrapper。


Rust 提供的接口说明
* 一个 TrtInfer struct
* 可以获取 ，输入 输出 特征的名字，类型
* allocate_memory_for(name, shape), 这里会分配 CPU(pinned) 和 GPU 内存 然后同时将device memory 绑定到 context 上
* get_pinned_memory(name) -> ndarray
* get_pinned_memory_mut(name)-> ndarray
* infer(). 进行推理。最终会调用 sync 等待结果写到 cpu 内存上

整体的用法说明：
* 状态初始化
    * 构建一个 TrtInfer 对象
    * 调用 allocate_memory_for 给 IO 分配 cpu 和 gpu 内存
* infer 过程
    * get_pinned_memory_mut(name)获取 ndarray
    * 往 ndarray 中塞一个batch 的数据
    * infer()
    * get_pinned_memory(name) 获取输出结果的 ndarray

其它：
* 由调用者保证是 static shape
* 目前模型大小就10多M，线程也不会开太多，所以先不用考虑 共享 context 的事情
* 注意 lengths 不是 shape inference io
* get_pinned_memory[_mut] 按照不同的 dtype 各实现一个方法. 
    1. 全返回 1D 数据，由调用者进行 reshpae
    2. 类型包含 f32, f16, bf16, int64, int32
* shape 由调用者传入