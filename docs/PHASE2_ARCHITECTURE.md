# Phase 2: Federated Foundation

Tài liệu này mô tả phần nền tảng Phase 2 đã được triển khai ngày 2026-07-24.
Mục tiêu thiết kế là giữ thuật toán federated, metric và Flower runtime hoạt
động ngay cả khi toàn bộ pipeline Phase 1 bị thay thế.

## Trust boundary

Phase 1 được xem là một upstream chưa được xác minh. Phase 2 không import
`src.preprocess`, `src.graph_build` hoặc `src.model` trong core. Chỉ adapter
IoT-23 được phép biết các field PyG và cách khởi tạo model cũ.

```text
Phase 1 hiện tại hoặc pipeline mới
              │
              ▼
      FederatedTask adapter
              │  schema + named parameters + sufficient statistics
              ▼
Contracts ── Core FedAvg/metrics ── Flower ClientApp/ServerApp
              ▲
              │
       Toy task độc lập
```

Nếu Phase 1 refactor, chỉ adapter và bước tạo artifact cần đổi. Core aggregation,
global metrics và Flower boundary không được phụ thuộc vào class path, pickle,
PyG `Data`, hay tên hàm cụ thể của Phase 1.

## Các lớp bảo vệ

- `FederatedTask` là plugin contract duy nhất mà core và Flower sử dụng. Task
  phải cung cấp danh sách client, schema, initial named state, local train,
  local evaluate và metadata.
- `FeatureSchema`, `LabelSchema`, `GraphSchema` và `ModelSpec` có digest ổn định.
  Sai thứ tự feature, label mapping, parameter key, shape hoặc dtype sẽ fail
  trước aggregation.
- `ContractBundle` lưu JSON/NPZ và SHA-256. Bundle không pickle đối tượng
  `Preprocessor`, vì class path của pickle sẽ vỡ khi Phase 1 refactor.
- FedAvg dùng `num_examples` làm trọng số và từ chối state không tương thích.
  Tensor không phải số thực chỉ được chấp nhận khi mọi client gửi cùng giá trị.
- Global macro-F1 được tính từ tổng confusion matrix K x K, không lấy trung
  bình các client macro-F1. Những class vắng mặt vẫn nằm trong label schema cố
  định và nhận F1 bằng 0.
- Optimizer là local, được khởi tạo lại cho mỗi train message. FedProx dùng cùng
  task contract và thêm proximal term; core không giữ optimizer state giữa các
  round.

## Cấu trúc triển khai

- `src/federated/contracts/`: schema, task protocol, portable artifact bundle.
- `src/federated/core/`: named-state conversion, strict weighted FedAvg,
  confusion-matrix metrics và in-process runner không phụ thuộc Flower.
- `src/federated/adapters/toy.py`: hai client non-IID xác định, dùng để chứng
  minh core độc lập với Phase 1 và PyG.
- `src/federated/adapters/phase1_iot23.py`: anti-corruption adapter cho graph
  và E-GraphSAGE hiện tại.
- `src/federated/flower/`: generic `ClientApp`/`ServerApp`, FedAvg/FedProx và
  callback tổng hợp global metrics theo Message API của Flower 1.32.1.

## Chạy proof độc lập

Tạo môi trường riêng cho Phase 2; lệnh này không cài `torch-geometric`:

```bash
python -m venv .venv-phase2
source .venv-phase2/bin/activate
pip install -r requirements-phase2.txt
python -m unittest discover -s tests/federated -v
flwr run . --stream
```

`pyproject.toml` mặc định chạy toy federation gồm 2 client, 3 round, full
participation và FedAvg. Để đổi sang FedProx, override `strategy=fedprox` và
`proximal-mu`; các default nằm trong `src/federated/flower/config.py`, do đó
direct API và Flower CLI dùng cùng hành vi.

## Nối Phase 1 hiện tại

Caller phải chuẩn bị graph đã split và truyền mọi contract một cách tường minh:

```python
from src.federated.adapters.phase1_iot23 import (
    Phase1IoT23Task,
    make_phase1_model_factory,
)

task = Phase1IoT23Task(
    client_graphs=scenario_graphs,
    feature_columns=preprocessor.feature_columns,
    class_to_idx=class_to_idx,
    model_factory=make_phase1_model_factory(
        model_name="egraphsage",
        cfg=model_config,
    ),
    model_hyperparameters=model_config,
    source_metadata={"source": "phase1-iot23"},
)

task.contract_bundle(preprocessor=preprocessor).write(contract_directory)
```

Adapter kiểm tra graph fields, dimensions, boolean masks, mask coverage,
non-overlap, label range và metadata mapping. Nó không tự fit preprocessing,
không tự khám phá feature order và không âm thầm sửa artifact không hợp lệ.

Để chạy adapter này với implementation cũ, cài thêm dependency Phase 1:

```bash
pip install -r requirements.txt
```

## Những gì đã và chưa được chứng minh

Đã được chứng minh tự động:

- contract bundle round-trip và phát hiện checksum/schema drift;
- strict weighted FedAvg, fixed-K global metrics, FedAvg và FedProx toy path;
- IoT-23 adapter train/evaluate qua public contract bằng graph double, không
  import PyG;
- Flower 1.32.1 boundary và federation thật 2 client x 3 round không failure.

Chưa được xem là bằng chứng:

- kết quả macro-F1 `0.8773` của Phase 1 chưa được reproduce;
- repository chưa có tracked preprocessor/model artifact đủ để tái tạo run;
- full IoT-23/PyG federation chưa chạy trong môi trường hiện tại;
- LOSO Phase 1 hiện train trên toàn bộ edge trước khi chọn validation subset,
  nên không dùng số LOSO cũ làm acceptance gate cho Phase 2.

## Hướng phát triển kế tiếp

1. Tạo một data-preparation command cho Phase 1 adapter, xuất contract bundle
   và client manifest bất biến từ mỗi scenario; tuyệt đối không fit scaler ở
   từng client.
2. Chạy một-client equivalence gate: cùng initial state, sample, optimizer và
   seed thì adapter local train phải khớp centralized reference trên train mask.
3. Chạy 6 scenario-aligned client với FedAvg trước; log per-round train/test
   confusion matrix, bytes upload/download và model-spec digest.
4. Sau đó mới thêm FedProx sweep trên đúng client manifests và seeds của FedAvg.
5. Chỉ sau khi hai baseline trên ổn định mới thiết kế IID/non-IID repartition,
   DP, secure aggregation hoặc Kubernetes deployment.

Phase 1 có thể được sửa song song, nhưng một artifact mới chỉ được Phase 2 nhận
khi vượt qua contract validation và equivalence gate; không cần sửa core hay
Flower runtime để thích nghi với refactor đó.
