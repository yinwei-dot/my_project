from __future__ import annotations

if __package__ in {None, ""}:
    from src.PaperPipeline import run_pipeline
    from src.infra import build_pipeline_arg_parser, create_runtime
else:
    from .src.PaperPipeline import run_pipeline
    from .src.infra import build_pipeline_arg_parser, create_runtime


def main() -> int:
    parser = build_pipeline_arg_parser()
    args = parser.parse_args()
    config, logger, _ = create_runtime(args)
    summaries = run_pipeline(config, logger)
    logger.info("运行完成，共处理 %s 篇文档。", len(summaries))
    return 0


if __name__ == "__main__":
    import logging as _logging
    import os as _os
    _code = main()
    _logging.shutdown()   # 刷新并关闭所有日志 handler
    _os._exit(_code)       # 强制退出，跳过 CUDA 非守护线程的等待