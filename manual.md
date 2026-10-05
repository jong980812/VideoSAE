# SAE feature가 무슨 컨셉인지 보는 법

명령 세 개면 그림이 나온다. 전부 레포 최상위 폴더(`sae_clean/`)에서 실행한다.

아래 예시는 전부 `--model videomaev2-vitb-k710distill --layer 7`로 적혀 있다. 이게 정확히 무엇인지:

- **모델**: `videomaev2-vitb-k710distill`은 이 레포에서 붙인 이름(모델 ID)이고, 정식 이름은 **VideoMAE V2 ViT-B**다. 영상을 넣으면 특징을 내는 Vision Transformer(Base 크기, 블록 12개)로, Kinetics-710으로 fine-tune한 giant 모델에서 distill한 공식 체크포인트 `OpenGVLab/VideoMAEv2-Base`(Hugging Face)를 쓴다. 분류 head는 없다. 입력은 16프레임, 224×224. (다른 모델은 맨 아래 표)
- **레이어**: 12개 블록 중 7번 블록의 출력 (0번부터 센다).
- **SAE**: 그 출력으로 학습한 SAE. `--weights_dir`를 안 주면 배포용 SAE인 `weights/sae/videomaev2-vitb-k710distill/l7`을 쓴다 (Matryoshka BatchTopK, 768 → 12,288 latent, k = 20).

어떤 모델·SAE로 뽑은 결과인지는 2번, 3번 명령의 출력과 그림 맨 위에 항상 같이 적힌다.

## 1. 통계 뽑기 (모델·레이어마다 한 번만, 2~5분)

```bash
python scripts/top_activations.py --model videomaev2-vitb-k710distill --layer 7 \
    --clips runs/sae_sweep/clips.json --split val4k --data_root runs/sae_sweep/k400_val
```

클립 4,000개를 돌려서 "latent마다 어떤 클립에서 가장 세게 켜지는지"를 저장한다.

## 2. 볼 latent 번호 고르기

```bash
python scripts/show_latent.py --model videomaev2-vitb-k710distill --layer 7
```

맨 위 네 줄에 어떤 모델·레이어·SAE인지 나오고, 그 아래 한 클래스에 쏠려 있는 latent 20개가 나온다. 여기서 번호를 고른다.

```
model: VideoMAE V2 ViT-B (distilled from the giant fine-tuned on Kinetics-710, head stripped)
checkpoint: OpenGVLab/VideoMAEv2-Base @ 78c337a   (model ID videomaev2-vitb-k710distill)
layer: output of block 7 of 0-11, 8 x 14 x 14 patches (time x height x width)
SAE: MatroyshkaBatchTopKSAE, 768 -> 12,288 latents, k = 20, from weights/sae/videomaev2-vitb-k710distill/l7
latent  8424:  60% playing chess                  fires on 0.13% of clips
latent 10340:  41% snorkeling                     fires on 0.43% of clips
```

(`60%` = 이 latent가 켜지는 양의 60%가 그 클래스에서 나온다는 뜻)

## 3. 그림 만들기

```bash
python scripts/show_latent.py --model videomaev2-vitb-k710distill --layer 7 --latent 10340
```

그림 파일이 생긴다: `results/videomaev2-vitb-k710distill/top_activations/l7_latent10340.png`

**VS Code 왼쪽 파일 목록에서 이 파일을 클릭하면 그림이 열린다.**

번호를 여러 개 주면 한 번에 여러 장 만든다: `--latent 8424 10340 4953`

---

## 그림 읽는 법

- **맨 위 여섯 줄**:
  1. latent 번호, 전체 클립 중 켜지는 비율, 어떤 클립 목록으로 뽑았는지
  2. `model`: 모델의 정식 이름
  3. `checkpoint`: Hugging Face 체크포인트와 리비전, 이 레포의 모델 ID
  4. `layer`: 몇 번 블록의 출력인지, 패치 격자 (시간 × 세로 × 가로. 그림 한 줄의 프레임 수가 이 시간 칸 수다)
  5. `SAE`: SAE 종류, 크기, 가중치 폴더 (내가 학습한 SAE면 여기에 그 폴더가 나온다)
  6. 이 latent가 많이 켜지는 클래스 5개
- **그 아래 한 줄 = 클립 하나.** 이 latent가 가장 세게 켜지는 클립부터 8개가 나온다. 줄 위 글자는 점수와 클립 이름이다.
- 한 줄 안에서 왼쪽 → 오른쪽이 시간 순서다.
- **빨간 칸 = 이 latent가 켜진 위치.** 진할수록 세게 켜진 것이다.

빨간 칸이 **공통으로 덮고 있는 것**이 그 latent의 컨셉이다.

예: latent 10340은 클래스로는 "snorkeling"이지만, 그림을 보면 빨간 칸이 사람이 아니라 물속 산호·바닥 무늬에 찍힌다. 즉 "스노클링"이 아니라 "물속 바닥 무늬" feature다. 이렇게 클래스 이름과 실제 컨셉이 다를 수 있으니 꼭 그림을 본다.

---

## 다른 경우

**다른 모델·레이어**: 세 명령의 `--model`, `--layer`만 똑같이 바꾼다.

**내가 학습한 SAE**: 1번에 `--weights_dir`(SAE 위치)와 `--out`(저장할 곳)을 주고, 3번에 같은 `--weights_dir`와 `--top`(1번의 `--out`)을 준다.

```bash
python scripts/top_activations.py --model videomaev2-vitb-k710distill --layer 7 \
    --weights_dir runs/sae_sweep/weights/btk --out runs/sae_sweep/top/btk.pt \
    --clips runs/sae_sweep/clips.json --split val4k --data_root runs/sae_sweep/k400_val

python scripts/show_latent.py --model videomaev2-vitb-k710distill --layer 7 --top runs/sae_sweep/top/btk.pt
python scripts/show_latent.py --model videomaev2-vitb-k710distill --layer 7 --top runs/sae_sweep/top/btk.pt \
    --weights_dir runs/sae_sweep/weights/btk --latent 123
```

그림은 `runs/sae_sweep/top/btk_latent123.png`에 생긴다.

**돌아가는지만 빨리 보기**: 1번에 `--limit 64 --out /tmp/test.pt`를 붙인다 (64클립만, 1분 안쪽).

**다른 GPU**: `--device cuda:3`

---

## 알아둘 것

- 목록(2번)에 없는 latent도 번호만 넣으면 그림이 나온다. 대부분의 latent는 특정 클래스와 상관없는 질감·모양·움직임이다.
- 점수는 클립 안에서의 최댓값이다. 화면 한 귀퉁이에서 잠깐만 켜져도 그 클립 점수는 높다.
- 클립이 클래스당 10개뿐이라, 몇 개 클립에서만 켜지는 latent의 `60%` 같은 숫자는 대략적인 값이다.
- `.env`의 `K400_VAL`이 맞게 잡힌 서버에서는 1번의 `--clips ... --data_root ...`를 빼도 된다. 그러면 val 전체(19,877클립)를 돈다.

## 모델 ID 표

`--model`에 넣는 이름과 그 정체. 배포용 SAE는 표의 레이어마다 하나씩 `weights/sae/<모델 ID>/l<레이어>`에 있다.

| `--model` | 어떤 모델인가 | 체크포인트 (Hugging Face) | `--layer` |
|---|---|---|---|
| `videomaev2-vitb-k710distill` | VideoMAE V2 ViT-B. Kinetics-710으로 fine-tune한 giant 모델에서 distill한 공식 ViT-B (분류 head 없음) | `OpenGVLab/VideoMAEv2-Base` | 0–11 |
| `vivit-b-16x2-kinetics400` | ViViT-B/16x2. Kinetics-400으로 supervised fine-tune (Google) | `google/vivit-b-16x2-kinetics400` | 0–10 |
| `vjepa2-vitl-fpc64-256` | V-JEPA 2 ViT-L. Meta의 self-supervised 모델, encoder만 사용 | `facebook/vjepa2-vitl-fpc64-256` | 0–23 |
| `videoprism-base-f16r288` | VideoPrism-B (16프레임, 288px). Google 공개본의 커뮤니티 PyTorch 포팅 | `sposiboh/videoprism-base-f16r288-pt` | 0–15 |
| `siglip-so400m-patch14-384` | SigLIP SO400M/14 (384px). 이미지-텍스트 모델이라 클립의 가운데 프레임 한 장만 본다 | `google/siglip-so400m-patch14-384` | 0–26 |

## (선택) 노트북으로 보기

`sae_usage.ipynb` 맨 아래 "What a latent responds to" 셀도 같은 그림을 보여준다. Jupyter가 설치돼 있어야 하고(`pip install ipykernel`), 첫 셀의 `os.chdir(...)` 경로와 `MODEL_ID`, `LAYER`를 맞춘 뒤 위에서부터 실행한다. 위의 명령으로 충분하면 안 써도 된다.
