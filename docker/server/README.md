# Self-hosted GPU runner (CUDA 12.8)

A containerized GitHub Actions runner with the CUDA toolkit (`nvcc`), cuDNN 9, and Rust,
so the `gpu.yml` workflow (`--features annotate,cuda-preprocess`) builds and runs.

## Host prerequisites

- NVIDIA driver 570.x (already present; `nvidia-smi` shows CUDA 12.8).
- Docker + NVIDIA Container Toolkit (exposes the GPU to `--gpus all`):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

- At least ~30 GB free disk (devel base + ORT download + CUDA build artifacts).

## Build

```bash
docker build -t ultralytics-gpu-runner:cuda12.8 docker/server
```

## Run

`runner.sh` wraps build/run/stop/restart. Mint a repo-scoped registration token (needs repo admin),
then launch once:

```bash
TOKEN=$(gh api -X POST /repos/ultralytics/inference/actions/runners/registration-token --jq .token)

./runner.sh run "$TOKEN"          # builds the image if missing, then starts the runner
./runner.sh status                # container state + nvidia-smi / nvcc
./runner.sh logs                  # follow the runner log
./runner.sh restart               # no token needed; reuses the registration
./runner.sh stop                  # stop but keep it registered
./runner.sh rm                    # tear the container down
```

Only `run` needs a token; the entrypoint configures once and persists the registration, so
restarts and reboots reconnect without a new token. The runner auto-adds `self-hosted, Linux, X64`;
the `rust-gpu-runner` label makes `runs-on: [self-hosted, Linux, X64, rust-gpu-runner]` in `gpu.yml`
schedule onto it.

Org-scoped instead: `RUNNER_URL=https://github.com/ultralytics RUNNER_GROUP=Default ./runner.sh run "$TOKEN"`,
with the token from `gh api -X POST /orgs/ultralytics/actions/runners/registration-token --jq .token`.

## Verify

```bash
docker exec gpu-runner-5 bash -lc 'nvidia-smi; nvcc --version'
```

Both must print (driver + toolkit). The runner then shows Idle in the org runner list.

## Notes

- Keep the base image on CUDA 12.x while the driver is 12.8; leave `ORT_CUDA_VERSION: "12"`.
- `runner.sh rm` deletes the container but does not deregister; the entry shows offline in the
  repo/org runner list until you remove it there (or relaunch with `run`, which `--replace`s it).
- TensorRT is not installed. The current tests only use `Device::Cuda`, so it is not needed. If
  you later add `Device::TensorRt` tests, add `RUN apt-get update && apt-get install -y tensorrt`
  to the Dockerfile (~1.5 GB; pulls TensorRT 10 from the CUDA repo already configured in the base).
