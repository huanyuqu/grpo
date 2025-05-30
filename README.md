1. python3.8.10安装trl没有GRPO Trainer
2. 使用vllm时需要先启动vllm server (如果用A100/30系以前的显卡需要加--dtype half), 最好指定一下TP/PP, vllm可能会根据available的GPUs自动设置, 但小模型的话也可能不会, 所以需要显式指定一下
```
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.api_server --model <your_model_name_or_path> --host 0.0.0.0 --port 8000 --dtype half --data-parallel-size 2
```
或者
```
trl vllm-serve --model <your_model_name_or_path> --host 0.0.0.0 --port 8000
```
3. 还是没启动成功vllm, 报错:
```
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/qhy/miniconda3/lib/python3.9/site-packages/starlette/datastructures.py", line 668, in __getattr__
[rank0]:     return self._state[key]
[rank0]: KeyError: 'engine_client'
...
[rank0]:   File "/home/qhy/miniconda3/lib/python3.9/site-packages/starlette/datastructures.py", line 671, in __getattr__
[rank0]:     raise AttributeError(message.format(self.__class__.__name__, key))
[rank0]: AttributeError: 'State' object has no attribute 'engine_client'
```

4. 这个版本的vllm有bug, 要在`vllm/entrypoints/api_server.py`里面加一行:
```
async def init_app(
    ...
    engine = (llm_engine
              if llm_engine is not None else AsyncLLMEngine.from_engine_args(
                  engine_args, usage_context=UsageContext.API_SERVER))

    app.state.engine_client = engine
    return app
```
5. 可以通过vllm_mode设置成colocate让generation和training的过程在相同的GPU上, 但最新的releasev0.17.0还没有支持这个功能

6. 用deepspeed之前需要安装CUDA Toolkit

7. `python -m accelerate.commands.launch --config_file <ACCELERATE_WITH_DEEPSPEED_CONFIG_FILE.yaml> train_grpo.py` 可以避免使用系统自带python的accelerate

8. Deepspeed的配置里面要注意num_processes和GPU数量是对应的