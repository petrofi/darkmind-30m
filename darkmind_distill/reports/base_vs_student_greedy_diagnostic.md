# Base vs Student Greedy Diagnostic

Decoding is deterministic greedy argmax. No sampling, no top-p, no multinomial sampling.
Device: `cuda`
Settings: `max_new_tokens=80`, `argmax=True`

## Summary

- base: failures `8/8`, true-flag counts `{'mixed_or_foreign_script': 2, 'gibberish': 8, 'language_match': 6}`
- student: failures `7/8`, true-flag counts `{'mixed_or_foreign_script': 4, 'gibberish': 7, 'language_match': 2, 'prompt_relevant': 1}`

## `tr_hello`

Prompt: `Merhaba.`

### base
- Model input: `Kullanıcı: Merhaba.
Asistan:`
- Generated token IDs: `[225, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152, 257, 152]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': True, 'gibberish': True, 'language_match': False, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
מממממממממממממממממממממממממממממממממממממממ�
```

### student
- Model input: `Kullanıcı: Merhaba.
Asistan:`
- Generated token IDs: `[225, 68, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': True, 'gibberish': True, 'language_match': False, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
`ווווווווווווווווווווווווווווווווווווווו
```

## `tr_turkey`

Prompt: `Türkiye hakkında iki cümle yaz.`

### base
- Model input: `Kullanıcı: Türkiye hakkında iki cümle yaz.
Asistan:`
- Generated token IDs: `[382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': False, 'gibberish': True, 'language_match': True, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türki
```

### student
- Model input: `Kullanıcı: Türkiye hakkında iki cümle yaz.
Asistan:`
- Generated token IDs: `[382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485, 588, 322, 73, 16, 382, 371, 485]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': False, 'gibberish': True, 'language_match': True, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türkiye'de, Türki
```

## `tr_python`

Prompt: `Python nedir?`

### base
- Model input: `Kullanıcı: Python nedir?
Asistan:`
- Generated token IDs: `[203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967, 30, 203, 3967]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': False, 'gibberish': True, 'language_match': True, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri:
Veri
```

### student
- Model input: `Kullanıcı: Python nedir?
Asistan:`
- Generated token IDs: `[225, 68, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': True, 'gibberish': True, 'language_match': False, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
`ווווווווווווווווווווווווווווווווווווווו
```

## `tr_docker`

Prompt: `Docker nedir?`

### base
- Model input: `Kullanıcı: Docker nedir?
Asistan:`
- Generated token IDs: `[428, 1128, 79, 265, 5, 203, 40, 1128, 79, 265, 5, 203, 40, 1128, 79, 265, 5, 203, 40, 1128, 79, 265, 5, 203, 40, 1128, 79, 265, 5, 203, 40, 1128, 79, 265, 5, 203, 40, 1128, 79, 265, 5, 203, 40, 1128, 79, 265, 5, 203, 40, 1128, 79, 265, 5, 203, 40, 1128, 79, 265, 5, 203, 40, 1128, 79, 265, 5, 203, 40, 1128, 79, 265, 5, 203, 40, 1128, 79, 265, 5, 203, 40, 1128]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': False, 'gibberish': True, 'language_match': True, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
Docker!
Docker!
Docker!
Docker!
Docker!
Docker!
Docker!
Docker!
Docker!
Docker!
Docker!
Docker!
Docker!
Doc
```

### student
- Model input: `Kullanıcı: Docker nedir?
Asistan:`
- Generated token IDs: `[428, 1128, 79, 265, 5, 203, 2]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': False, 'gibberish': False, 'language_match': True, 'prompt_relevant': True, 'identity_correct': False, 'identity_leakage': False}`

```text
Docker!
```

## `tr_identity`

Prompt: `Sen kimsin?`

### base
- Model input: `Kullanıcı: Sen kimsin?
Asistan:`
- Generated token IDs: `[225, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': True, 'gibberish': True, 'language_match': False, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
כככככככככככככככככככככככככככככככככככככככ�
```

### student
- Model input: `Kullanıcı: Sen kimsin?
Asistan:`
- Generated token IDs: `[225, 68, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248, 152, 248]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': True, 'gibberish': True, 'language_match': False, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
`ווווווווווווווווווווווווווווווווווווווו
```

## `en_hello`

Prompt: `Hello.`

### base
- Model input: `Kullanıcı: Hello.
Asistan:`
- Generated token IDs: `[802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16, 802, 353, 563, 16]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': False, 'gibberish': True, 'language_match': True, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello, Hello,
```

### student
- Model input: `Kullanıcı: Hello.
Asistan:`
- Generated token IDs: `[802, 353, 563, 91, 87, 16, 225, 57, 2221, 709, 515, 93, 16, 225, 57, 2221, 709, 515, 93, 16, 225, 57, 2221, 709, 515, 93, 16, 225, 57, 2221, 709, 515, 93, 16, 225, 57, 2221, 709, 515, 93, 16, 225, 57, 2221, 709, 515, 93, 16, 225, 57, 2221, 709, 515, 93, 16, 225, 57, 2221, 709, 515, 93, 16, 225, 57, 2221, 709, 515, 93, 16, 225, 57, 2221, 709, 515, 93, 16, 225, 57, 2221, 709]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': False, 'gibberish': True, 'language_match': False, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
Hellows, University, University, University, University, University, University, University, University, University, University, Univer
```

## `en_python`

Prompt: `What is Python?`

### base
- Model input: `Kullanıcı: What is Python?
Asistan:`
- Generated token IDs: `[2599, 1162, 894, 225, 91, 301, 225, 91, 2645, 225, 91, 4016, 3383, 73, 225, 57, 2221, 709, 515, 93, 2931, 409, 76, 86, 1075, 77, 263, 409, 76, 86, 1075, 77, 263, 409, 76, 86, 1075, 77, 263, 409, 76, 86, 1075, 77, 263, 409, 76, 86, 1075, 77, 263, 409, 76, 86, 1075, 77, 263, 409, 76, 86, 1075, 77, 263, 409, 76, 86, 1075, 77, 263, 409, 76, 86, 1075, 77, 263, 409, 76, 86, 1075, 77]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': False, 'gibberish': True, 'language_match': True, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
What is was well with the University of Christian Christian Christian Christian Christian Christian Christian Christian Christian Christi
```

### student
- Model input: `Kullanıcı: What is Python?
Asistan:`
- Generated token IDs: `[225, 68, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254, 152, 254]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': True, 'gibberish': True, 'language_match': False, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
`כככככככככככככככככככככככככככככככככככככככ
```

## `en_sentence`

Prompt: `Write one English sentence.`

### base
- Model input: `Kullanıcı: Write one English sentence.
Asistan:`
- Generated token IDs: `[382, 76, 73, 1056, 82, 75, 295, 87, 76, 1056, 82, 75, 295, 87, 76, 1056, 82, 75, 295, 87, 76, 1056, 82, 75, 295, 87, 76, 1056, 82, 75, 295, 87, 76, 1056, 82, 75, 295, 87, 76, 1056, 82, 75, 295, 87, 76, 1056, 82, 75, 295, 87, 76, 1056, 82, 75, 295, 87, 76, 1056, 82, 75, 295, 87, 76, 1056, 82, 75, 295, 87, 76, 1056, 82, 75, 295, 87, 76, 1056, 82, 75, 295, 87]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': False, 'gibberish': True, 'language_match': True, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
The English English English English English English English English English English English English Englis
```

### student
- Model input: `Kullanıcı: Write one English sentence.
Asistan:`
- Generated token IDs: `[225, 68, 70, 901, 79, 11, 16, 225, 68, 70, 901, 79, 11, 16, 225, 68, 70, 901, 79, 11, 16, 225, 68, 70, 901, 79, 11, 16, 225, 68, 70, 901, 79, 11, 16, 225, 68, 70, 901, 79, 11, 16, 225, 68, 70, 901, 79, 11, 16, 225, 68, 70, 901, 79, 11, 16, 225, 68, 70, 901, 79, 11, 16, 225, 68, 70, 901, 79, 11, 16, 225, 68, 70, 901, 79, 11, 16, 225, 68, 70]`
- Flags: `{'empty': False, 'mixed_or_foreign_script': False, 'gibberish': True, 'language_match': False, 'prompt_relevant': False, 'identity_correct': False, 'identity_leakage': False}`

```text
`back', `back', `back', `back', `back', `back', `back', `back', `back', `back', `back', `b
```

