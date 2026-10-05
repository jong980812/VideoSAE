## SAE classification 성능 구하기

```
# 레이어 9번에 SAE 넣었을 때
python scripts/evaluate.py --model vivit-b-16x2-kinetics400 --layer 9

# SAE 없이 eval 하기
python scripts/evaluate.py --model vivit-b-16x2-kinetics400 
```

모델 config는 [여기(config/vivit-b-16x2-kinetics400.yaml)](config/vivit-b-16x2-kinetics400.yaml)있음.  
여기에 SAE weight 위치나,  classification head 위치도 적혀 있음. 만약 다른 SAE weight을 쓰고 싶으면 config 파일 새로 만들어서 쓰면 됨. 

## SAE classification head

```
python scripts/train_probe.py --model videomaev2-vitb-k710distill
```

이런 식으로 하면 댐. 근데 classification head는 이미 다 트레이닝 했으니까 왠만하면 그대로 나두는 게 좋을듯함.  
OOD 때문에 classification head 따로 트레이닝한건 여기 코드 베이스에 안넣었음.

