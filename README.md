# SVD Transformer

An attempt to overcome the quadratic complexity of the classic transformer attention mechanism through SVD compression of the input matrices.

## Development practices

To setup the automated code formatting and lint script run `pip install pre-commit && pre-commit install`.
This will run [`pre-commit`](https://pre-commit.com/) each time you do a commit.
If you want to run `pre-commit` yourself do `pre-commit run -a`.

*NB*: Do not forget about `.pre-commit-config.yaml`.
