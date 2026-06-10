# `exact` package

`src/exact/` là package chính của hệ thống EXACT 2026. Code trong package này phục vụ pipeline dự đoán chính thức: nhận input từ dataset/API, route sang Type 1 hoặc Type 2, chạy pipeline tương ứng, rồi trả về response đúng format cuộc thi.

Mục tiêu tổ chức code là giữ ranh giới rõ giữa:

- phần dùng chung cho toàn hệ thống;
- phần xử lý dữ liệu/dataset;
- router điều phối request;
- pipeline Type 1 logic;
- pipeline Type 2 physics;
- solver symbolic;
- app/script entrypoint.

## Luồng xử lý tổng quát

```text
raw JSON/CSV or API payload
        │
        ▼
common.schemas.PredictionRequest
        │
        ▼
router.TaskRouter
        │
        ├── Type 1 logic   -> logic.pipeline.run_type1_pipeline()
        │
        └── Type 2 physics -> type2.pipeline.run_type2_pipeline()
        │
        ▼
common.schemas.PredictionResponse
        │
        ▼
to_official_response() / API response / prediction file
```

Rule chính:

> Các pipeline không nên xử lý raw JSON/CSV trực tiếp. Raw data phải được normalize thành schema chung trước.

## Cấu trúc folder hiện tại

```text
src/exact/
├── app/                 # FastAPI application entrypoint và API routes
├── baselines/           # Notebook baseline/thử nghiệm, không phải production core
├── common/              # Schema/contract dùng chung cho cả Type 1, Type 2, API, router
├── datasets/            # Loader và normalizer cho raw dataset EXACT
├── logic/               # Type 1 educational logic QA pipeline
├── prompts/             # Prompt template dùng cho LLM translation/generation
├── router/              # Task/question router trước khi gọi pipeline cụ thể
├── scripts/             # CLI eval/deployment helpers (không chạy batch inference)
├── symbolic_solvers/    # Solver backend dùng cho symbolic reasoning
├── type2/               # Type 2 physics pipeline boundary
├── config.py            # Runtime settings/env config dùng chung
├── llm_client.py        # Client gọi LLM/OpenAI-compatible endpoint
├── logger.py            # Logging setup dùng chung
└── README.md            # Tài liệu tổ chức package này
```

## Các module dùng chung

### `common/`

`common/` chứa contract dùng chung giữa nhiều phần của hệ thống. Đây là nơi đặt các object không thuộc riêng Type 1, Type 2, dataset, hay app.

Hiện có:

```text
common/
├── __init__.py
└── schemas.py
```

`common/schemas.py` định nghĩa:

- `TaskType`: loại task cấp cao (`type1_logic`, `type2_physics`).
- `QuestionType`: dạng câu hỏi (`mcq`, `yes_no_uncertain`, `open_ended`, `numerical`).
- `PredictionRequest`: input chuẩn sau khi normalize.
- `PredictionResponse`: output chuẩn nội bộ và gần với format nộp bài.
- `to_official_response()`: convert response nội bộ sang shape chính thức (6 field nộp bài).

Quy tắc:

- Code mới nên import schema từ `exact.common.schemas`.
- Không thêm logic task-specific vào `common/`.
- Chỉ đưa vào `common/` khi module đó thật sự được dùng bởi nhiều phần độc lập.

Ví dụ đúng:

```python
from exact.common.schemas import PredictionRequest, PredictionResponse, QuestionType, TaskType
```

### `config.py`

Quản lý runtime settings/env config dùng chung: LLM provider/model, logging, token limit, endpoint, v.v.

Không nên đặt logic pipeline, loader, solver, hoặc prompt construction ở đây.

### `logger.py`

Quản lý logging setup và request-aware logger.

Dùng cho:

- API;
- batch scripts;
- Type 1 pipeline;
- Type 2 pipeline;
- LLM client;
- dataset loading.

Package code nên dùng logger thay vì `print()`.

### `llm_client.py`

Client gọi LLM thông qua OpenAI-compatible API. Đây là shared infrastructure vì Type 1 translator, future verifier, hoặc Type 2 extractor đều có thể cần LLM.

Không nên nhúng prompt/task logic vào file này. Prompt/task logic nên nằm ở module pipeline/translator tương ứng.

## Folder theo chức năng

### `app/`

FastAPI layer.

```text
app/
├── main.py      # Tạo FastAPI app
└── router.py    # /health, /predict, /batch
```

Vai trò:

- nhận request từ API;
- validate bằng `PredictionRequest`;
- gọi `TaskRouter`;
- dispatch sang `logic.pipeline` hoặc `type2.pipeline`;
- trả `PredictionResponse`.

Không nên:

- parse raw dataset;
- gọi solver trực tiếp;
- chứa business logic Type 1/Type 2.

### `datasets/`

Dataset ingestion layer.

```text
datasets/
├── dataset.py          # Map-style dataset wrapper, LoadedExample
├── loader.py           # Đọc/flatten/normalize raw JSON/CSV
├── schemas.py          # Compatibility shim re-export từ common.schemas
└── exact/              # Local dataset files
```

Vai trò:

- đọc raw EXACT JSON/CSV;
- normalize field name như `premises-NL`, `answers`, `unit`;
- tạo payload hợp lệ cho `PredictionRequest`;
- hỗ trợ train/dev/test split hoặc dataset exploration.

`datasets/schemas.py` chỉ còn là compatibility shim. Code mới không nên import từ đây nữa.

Không nên:

- chạy LLM;
- chạy solver;
- quyết định đáp án;
- chứa pipeline Type 1/Type 2.

### `router/`

Routing layer.

```text
router/
└── task_router.py
```

Vai trò:

- quyết định request là Type 1 hay Type 2;
- detect question shape cho Type 1: MCQ, Yes/No/Uncertain, Open-ended;
- trả `RouteDecision` gồm `task_type`, `question_type`, `reason`.

Rule:

- Task-level routing nằm ở đây, không nằm trong pipeline.
- Pipeline nhận `question_type` đã được router quyết định.
- Router không được chạy solver hoặc LLM.

### `logic/`

Type 1 educational logic QA pipeline.

```text
logic/
├── pipeline.py        # Type 1 orchestration, YNU + MCQ path
├── ir/                # Atom, Fact, Rule, Query, ProofStep, SolveResult, formula IR
├── parsing/           # Heuristic parser NL -> IR, FOL parser
│   ├── parser.py
│   └── fol_parser.py
├── translation/       # LLM semantic parser -> IR
│   ├── llm_translator.py
│   └── prompts.py
├── kb/                # KnowledgeBase construction/cache boundary
├── explain/           # Proof trace -> explanation/cot/premises
└── README.md          # Tài liệu riêng cho Type 1 logic framework
```

Vai trò:

- translate premises/question thành IR;
- build KB;
- gọi symbolic solver;
- xử lý Yes/No/Unknown và MCQ option evaluation;
- tạo explanation dựa trên proof trace.

Không nên:

- route Type 1 vs Type 2;
- đọc raw dataset file;
- chứa Type 2 physics logic.

### `symbolic_solvers/`

Solver backend layer.

```text
symbolic_solvers/
├── base.py
├── forward_chain/
│   └── solver.py
└── z3_solver/
    ├── encoder.py
    └── solver.py
```

Vai trò:

- cung cấp solver interface;
- thực thi symbolic reasoning trên IR;
- giữ proof/provenance phục vụ explanation.

Hiện tại:

- `forward_chain/solver.py`: solver mặc định cho Horn-style rules, đã hỗ trợ unification.
- `z3_solver/`: backend thử nghiệm Boolean/Z3.

Rule:

- Solver không nên biết API/dataset format.
- Solver nhận `KnowledgeBase` + `Atom`, trả `SolveResult`.
- Solver không gọi LLM.

### `type2/`

Type 2 physics pipeline boundary.

```text
type2/
└── pipeline.py
```

Hiện tại là placeholder ổn định API cho nhánh physics. Vì Type 2 do teammate phụ trách, folder này nên giữ độc lập với Type 1.

Rule:

- Không import logic-specific IR nếu không thật sự cần.
- Dùng schema chung từ `common.schemas`.
- Khi implement thật, nên tách quantity extraction, formula selection, execution, unit conversion, verification thành module con nếu code lớn lên.

### `prompts/`

Prompt template layer.

```text
prompts/
└── prompts.py
```

Vai trò:

- lưu prompt template có thể tái sử dụng;
- tránh hard-code prompt dài rải rác trong pipeline.

Rule:

- Prompt chung có thể nằm ở đây.
- Prompt rất đặc thù cho một translator/pipeline có thể nằm cạnh module đó nếu dễ maintain hơn.

### `scripts/`

Helper CLIs for offline evaluation and deployment prep. Production inference
uses the FastAPI service (`POST /predict`), not batch runners.

```text
scripts/
├── config_utils.py
├── evaluate_type1_predictions.py
├── evaluate_type2_predictions.py
└── pull_model.py
```

Vai trò:

- `evaluate_*_predictions.py`: chấm điểm file JSON đã lưu (sau khi gọi API).
- `pull_model.py`: tải/cache model Hugging Face trước khi chạy vLLM.

### `baselines/`

Notebook baseline và walkthrough.

```text
baselines/
├── B01_zero_shot.ipynb
├── B02_unit_aware_pot.ipynb
├── B03_current_pipeline_walkthrough.ipynb
└── B03_kaggle_end_to_end_pipeline.ipynb
```

Vai trò:

- experiment;
- analysis;
- demo pipeline;
- baseline comparison.

Rule:

- Notebook không phải source of truth.
- Nếu logic trong notebook cần dùng production, hãy đưa vào module `.py` tương ứng.
- Notebook có thể import production code, nhưng production code không import notebook.

## Quy tắc tổ chức code

### 1. Shared code đặt ở `common/`, nhưng không lạm dụng

Chỉ đưa vào `common/` khi code/schema được dùng bởi nhiều vùng độc lập.

Nên đặt ở `common/`:

- request/response schema;
- enum dùng chung;
- official response conversion;
- helper chung thật sự dùng bởi cả Type 1 và Type 2.

Không nên đặt ở `common/`:

- Type 1 IR;
- KB/proof logic;
- Type 2 formula solver;
- dataset-specific parser;
- prompt đặc thù một task.

### 2. Raw dataset chỉ thuộc `datasets/`

Các key như `premises-NL`, `premises-FOL`, `answers`, `unit`, CSV column name nên được xử lý trong `datasets/loader.py` hoặc `datasets/dataset.py`.

Sau khi ra khỏi dataset layer, code nên dùng `PredictionRequest`.

### 3. Router chỉ route, không reason

`router/` được quyền inspect shape của request để quyết định:

- Type 1 vs Type 2;
- MCQ vs Yes/No/Uncertain vs Open-ended;
- Numerical cho Type 2.

Router không được:

- gọi LLM;
- chạy solver;
- chọn đáp án;
- build explanation.

### 4. Pipeline điều phối, solver giải quyết

Pipeline chịu trách nhiệm orchestration:

- gọi translator/parser;
- build KB;
- gọi solver;
- convert result thành response.

Solver chịu trách nhiệm reasoning:

- derive facts;
- prove claim/negation;
- trả proof trace.

Không nhét solver algorithm trực tiếp vào API/script/router.

### 5. Type 1 và Type 2 độc lập

Type 1 nằm trong `logic/`.
Type 2 nằm trong `type2/`.

Hai nhánh chỉ nên giao tiếp qua:

- `common.schemas`;
- `router`;
- shared infra như `logger`, `config`, `llm_client`.

### 6. Compatibility shim được phép, nhưng code mới dùng path mới

`datasets/schemas.py` tồn tại để notebook hoặc code cũ không gãy.

Code mới phải dùng:

```python
from exact.common.schemas import PredictionRequest
```

Không dùng:

```python
from exact.datasets.schemas import PredictionRequest
```

### 7. Import style

Dùng absolute import trong package:

```python
from exact.common.schemas import PredictionResponse
from exact.logic.pipeline import run_type1_pipeline
```

Tránh relative import sâu hoặc import từ source-root:

```python
from config import get_settings  # không dùng
```

### 8. Không để artifact trong package

Không đưa vào `src/exact/`:

- prediction output;
- logs;
- checkpoint;
- generated artifact;
- temporary files.

Các thứ đó nên nằm ở:

```text
outputs/
artifacts/
```

### 9. Test theo behavior, không test implementation detail quá mức

Nên có focused tests cho:

- router detection;
- MCQ option parsing/winner policy;
- forward-chain unification;
- schema compatibility;
- Type 1/Type 2 response shape.

Không cần test private helper nếu behavior public đã cover đủ.

## Khi thêm module mới

Trước khi tạo folder/file mới, tự hỏi:

1. Code này thuộc Type 1, Type 2, dataset, router, app, hay shared?
2. Nó có được dùng bởi nhiều nhánh không, hay chỉ một pipeline?
3. Nó có phụ thuộc raw dataset format không?
4. Nó có gọi LLM/solver/API không?
5. Có thể test bằng input/output nhỏ không?

Nếu chưa rõ, ưu tiên đặt gần nơi dùng nhất. Chỉ move lên `common/` khi có nhu cầu dùng chung thật sự.
