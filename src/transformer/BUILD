python_sources(
    dependencies=[
        ":training_data",
    ],
    run_goal_use_sandbox=False,
)

resources(
    name="training_data",
    sources=[
        "tiny_shakespeare.txt",
        "training_data.txt",
        "*.eqx",
        "*.pkl",
    ],
)