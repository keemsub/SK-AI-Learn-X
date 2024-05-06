features = []  # 과거 48주간의 가격
for i in range(48):
    df_train[f"price-{i+1}week"] = df_train["철근 고장력 HD10mm 현물KRW/ton"].shift(
        i + 1
    )
    features.append(f"price-{i+1}week")

targets = []  # 미래 48주간의 가격
for i in range(48):
    df_train[f"price+{i+1}week"] = df_train["철근 고장력 HD10mm 현물KRW/ton"].shift(
        -(i + 1)
    )
    targets.append(f"price+{i+1}week")

train = df_train.dropna(axis=0)

train_x = train[features]
train_y = train[targets]
# model.fit(train[features], train[targets])
