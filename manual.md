# SAE feature가 무슨 컨셉인지 보는 법

두 단계다.

1. **뽑기**: `scripts/top_activations.py`로 클립들을 한 번 돌려서, latent마다 "가장 세게 켜지는 클립"과 "어느 클래스에서 켜지는지"를 파일 하나로 저장한다.
2. **보기**: `sae_usage.ipynb` 맨 아래 셀에서 그 파일을 읽어 그림으로 본다.

명령은 전부 레포 최상위 폴더(`sae_clean/`)에서 실행한다.

---

## 1. 뽑기

### 배포용 SAE (`weights/sae/`)로 뽑기

```bash
python scripts/top_activations.py --model videomaev2-base --layer 7 \
    --clips runs/sae_sweep/clips.json --split val4k --data_root runs/sae_sweep/k400_val
```

- 결과: `results/videomaev2-base/top_activations/l7.pt` (약 21MB)
- 걸리는 시간: GPU 한 장으로 2~5분 (4,000클립)
- `--model`, `--layer`만 바꾸면 다른 모델·레이어도 똑같이 된다.

> `--clips ... --data_root ...` 세 개는 이 서버에 있는 Kinetics val(4,000클립, 클래스당 10개)을 쓰라는 뜻이다.
> `.env`의 `K400_VAL`이 맞게 잡혀 있는 서버라면 이 세 개를 빼도 되고, 그러면 val 전체(19,877클립)를 돈다.

### 내가 학습한 SAE로 뽑기

`--weights_dir`로 SAE 위치를, `--out`으로 저장할 곳을 준다. (`--out`을 안 주면 위의 배포용 결과를 덮어쓴다.)

```bash
python scripts/top_activations.py --model videomaev2-base --layer 7 \
    --weights_dir runs/sae_sweep/weights/btk --out runs/sae_sweep/top/btk.pt \
    --clips runs/sae_sweep/clips.json --split val4k --data_root runs/sae_sweep/k400_val
```

`--weights_dir` 안에는 `<모델 ID>/l<레이어>/ae.pt`가 있어야 한다. `--layer`는 그 SAE를 학습한 레이어와 같아야 한다.

### 일단 돌아가는지만 빨리 보기

`--limit 64`를 붙이면 64클립만 돈다 (모델 로딩 포함 1분 안쪽). 결과는 의미 없으니 `--out`을 임시 경로로 준다.

```bash
python scripts/top_activations.py --model videomaev2-base --layer 7 --limit 64 --out /tmp/test.pt \
    --clips runs/sae_sweep/clips.json --split val4k --data_root runs/sae_sweep/k400_val
```

### 다른 GPU 쓰기

`--device cuda:3`

---

## 2. 노트북에서 보기 (`sae_usage.ipynb`)

### 준비

- 첫 셀의 `os.chdir(...)` 경로를 이 레포 위치로 바꾼다.
- `MODEL_ID`, `LAYER`가 있는 셀에서 두 값을 1번에서 뽑은 것과 똑같이 맞춘다.
- 위에서부터 차례로 실행한다. 맨 아래 **"What a latent responds to"** 부분이 새로 추가된 셀이다.

### 볼 만한 latent 목록 보기

"What a latent responds to" 아래 첫 번째 코드 셀을 실행하면 한 클래스에 가장 쏠려 있는 latent 20개가 나온다.

```
latent  8424:  60% playing chess                  fires on 0.13% of clips
latent  3956:  48% swimming breast stroke         fires on 0.23% of clips
```

- `60%`: 이 latent의 activation 중 60%가 그 클래스에서 나온다.
- `fires on 0.13% of clips`: 전체 클립 중 이 latent가 켜지는 비율.

### latent 하나 자세히 보기

그 다음 셀에서 `LATENT = 8424`처럼 번호를 넣고 실행한다. (그대로 두면 목록 1등을 보여준다.)

- 위쪽 글자: 이 latent가 많이 켜지는 클래스 5개
- 아래쪽 그림: 가장 세게 켜지는 클립 8개. 한 줄이 클립 하나이고, **빨간 칸이 latent가 켜진 위치**다 (진할수록 세게).

### 내가 학습한 SAE를 볼 때

두 줄만 바꾼다.

```python
# "Load the model and the layer's SAE" 셀
sae = load_sae(MODEL_ID, LAYER, device=DEVICE, backbone=videomodel, weights_dir="runs/sae_sweep/weights/btk")

# "What a latent responds to" 첫 번째 코드 셀
TOP = Path("runs/sae_sweep/top/btk.pt")
```

SAE와 결과 파일이 서로 안 맞으면 `was computed with another SAE` 에러가 난다. 둘을 같은 SAE로 맞추면 된다.

---

## 3. 결과 파일 직접 열어보기

노트북 없이 숫자만 볼 때.

```python
import torch
top = torch.load("results/videomaev2-base/top_activations/l7.pt")

L = 8424
for score, i in zip(top["top_val"][L].tolist(), top["top_clip"][L].tolist()):
    if i >= 0:
        print(f"{score:.2f}  {top['keys'][i]}")
```

| 이름 | 모양 | 뜻 |
|---|---|---|
| `top_val` | (latent 수, 8) | latent별 상위 8개 클립의 점수 |
| `top_clip` | (latent 수, 8) | 그 클립의 번호 (`keys`의 인덱스, 없으면 -1) |
| `class_mean` | (latent 수, 클래스 수) | 클래스별 평균 점수 |
| `clip_freq` | (latent 수,) | latent가 켜지는 클립의 비율 |
| `keys`, `labels`, `classes` | | 클립 이름, 클립의 클래스 번호, 클래스 이름 |

---

## 읽을 때 알아둘 것

- **클래스 이름만 믿지 말고 그림을 본다.** 예: latent 8424는 "playing chess"로 나오지만, 그림을 보면 체스판 칸과 휴대폰 자판에서 켜진다. 실제로는 "작은 네모 격자 무늬"에 가깝다.
- **점수는 클립 안에서의 최댓값이다.** 화면 한 귀퉁이에서 잠깐만 켜져도 그 클립 점수는 높다.
- **대부분의 latent는 특정 클래스와 상관없다.** 질감, 모양, 움직임 같은 것일 수 있다. 목록에 안 나오는 latent도 번호를 넣으면 볼 수 있다.
- **클립 수가 적으면 우연이 섞인다.** val4k는 클래스당 10개뿐이라, 몇 개 클립에서만 켜지는 latent의 `60%` 같은 숫자는 대략적인 값이다.
