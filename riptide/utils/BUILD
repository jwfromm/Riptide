py_binary(
    name = "datasets",
    srcs = ["datasets.py"],
)

py_test(
    name = "datasets_test",
    srcs = ["datasets_test.py"],
    deps = [
        ":datasets",
        "//models/research/slim:inception_preprocessing",
    ],
)
