# DarkMind v2 Phase 2C Tiny Full-Epoch Samples

These 40 examples are selected deterministically across quality bands from the final 200-prompt greedy audit.
`Relative stronger` means stronger only within this weak checkpoint; it is not a fluency claim. No output was edited or sanitized.
Continuations with trailing whitespace are shown as JSON strings so the boundary whitespace remains explicit without invalid Markdown line endings.

## Review Summary

- Basic grammatical structure: occasional short fragments, especially in English; not reliable.
- Topical continuation: generally absent.
- Language consistency: strong for greedy decoding, but content quality remains weak.
- Lexical diversity: poor under greedy decoding; seeded sampling is more diverse but still unreliable.
- Sentence completion: two short Turkish completions are minimally plausible; most outputs loop or terminate as fragments.
- Factual reliability: no reviewed factual continuation is usable.
- Code usefulness: no reviewed technical/code continuation is usable.

Checkpoint: `step_002867_tokens_011743232`; decoding: greedy, EOS-aware, max 32 new tokens.

## Turkish ordinary continuations

### 1. tr_ordinary_009 - relative stronger

Prompt:
```text
Yağmurdan sonra küçük park günlük yaşamın küçük bir parçası olarak
```
Raw continuation:
```text
 kabul edilir.
```
Warnings: `none`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 2. tr_ordinary_028 - relative stronger

Prompt:
```text
Yaz akşamında balkon sessiz görünüyordu çünkü
```
Raw continuation:
```text
 bu yüzden bir şey var.
```
Warnings: `none`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 3. tr_ordinary_016 - relative stronger

Prompt:
```text
Uzun toplantının ardından ofis sessiz görünüyordu çünkü
```
Raw continuation:
```text
 bu kadar çok sayıda bir şekilde, bu, bu, bu, bu, bu, bu, 1999999999'de
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.3750`; EOS: `False`.

### 4. tr_ordinary_025 - relative stronger

Prompt:
```text
Kış sabahında mutfak sessiz görünüyordu çünkü
```
Raw continuation:
```text
 bu kadar çok sayıda bir şekilde, bu, bu, bu, bu, bu, bu, 1999999999'da
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.3750`; EOS: `False`.

### 5. tr_ordinary_049 - relative stronger

Prompt:
```text
Öğle arasında küçük kafe sessiz görünüyordu çünkü
```
Raw continuation:
```text
" bu kadar çok sayıda bir şekilde, bu, bu, bu, bu, bu, bu, 199999999'de "
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.3750`; EOS: `False`.

### 6. tr_ordinary_014 - average

Prompt:
```text
İstasyonda bekleyen yolcular hakkında ilk fark edilen ayrıntı
```
Raw continuation:
```text
larını, 199999'da, 1'de 19999'de 1'de 199
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.2500`; EOS: `False`.

### 7. tr_ordinary_015 - average

Prompt:
```text
İstasyonda bekleyen yolcular günlük yaşamın küçük bir parçası olarak
```
Raw continuation:
```text
, bu nedenle, bu, bu, bu, bu, bu, 1999999999'da, 199
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.2500`; EOS: `False`.

### 8. tr_ordinary_017 - average

Prompt:
```text
Uzun toplantının ardından ofis hakkında ilk fark edilen ayrıntı
```
Raw continuation:
```text
ları, 199999'da, 1'da, 1999'da 1'de 199
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.2500`; EOS: `False`.

### 9. tr_ordinary_020 - average

Prompt:
```text
Pazar yerindeki eski dükkân hakkında ilk fark edilen ayrıntı
```
Raw continuation:
```text
ları, 199999'da, 1'de 19999'de 1'de 199
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.2500`; EOS: `False`.

### 10. tr_ordinary_023 - average

Prompt:
```text
Deniz kıyısındaki yürüyüş yolu hakkında ilk fark edilen ayrıntı
```
Raw continuation:
```text
ları, 199999'da, 1'da, 1999'da 1'de 199
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.2500`; EOS: `False`.

### 11. tr_ordinary_021 - weak

Prompt:
```text
Pazar yerindeki eski dükkân günlük yaşamın küçük bir parçası olarak
```
Raw continuation:
```text
, 1999999'de, 1'de 1999'de 1'de 1999
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.1875`; EOS: `False`.

### 12. tr_ordinary_036 - weak

Prompt:
```text
Şehir merkezindeki meydan günlük yaşamın küçük bir parçası olarak
```
Raw continuation:
```text
, 1999999'de, 1'de, 199999'de 1'de 1
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.1875`; EOS: `False`.

### 13. tr_ordinary_042 - weak

Prompt:
```text
Tren yolculuğundaki kompartıman günlük yaşamın küçük bir parçası olarak
```
Raw continuation:
```text
", 1999999'de 1999'de 199'de 1999'de "
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.1875`; EOS: `False`.

### 14. tr_ordinary_054 - weak

Prompt:
```text
Günün son otobüsü günlük yaşamın küçük bir parçası olarak
```
Raw continuation:
```text
, 19999'de, 1999'de 1'de 19999'de 1'
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.1875`; EOS: `False`.

### 15. tr_ordinary_019 - weak

Prompt:
```text
Pazar yerindeki eski dükkân sessiz görünüyordu çünkü
```
Raw continuation:
```text
 199999'de 1999'de 1'de 19999'de 1'de
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.1562`; EOS: `False`.
## English ordinary continuations

### 16. en_ordinary_001 - relative stronger

Prompt:
```text
Early in the morning the neighborhood seemed unusually quiet because
```
Raw continuation:
```text
 of the first-i.
```
Warnings: `none`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 17. en_ordinary_003 - relative stronger

Prompt:
```text
After the long meeting the office seemed unusually quiet because
```
Raw continuation:
```text
 of the first.
```
Warnings: `none`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 18. en_ordinary_005 - relative stronger

Prompt:
```text
When the rain stopped the small park seemed unusually quiet because
```
Raw continuation:
```text
 of the first-in.
```
Warnings: `none`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 19. en_ordinary_007 - relative stronger

Prompt:
```text
At the quiet end of the station seemed unusually quiet because
```
Raw continuation:
```text
 of the first-s.
```
Warnings: `none`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 20. en_ordinary_008 - relative stronger

Prompt:
```text
At the quiet end of the station first drew attention when
```
Raw continuation:
```text
 the "C.
```
Warnings: `none`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 21. en_ordinary_033 - average

Prompt:
```text
The garden after several dry days seemed unusually quiet because
```
Raw continuation:
```text
 of the first-in.
```
Warnings: `none`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 22. en_ordinary_034 - average

Prompt:
```text
The garden after several dry days first drew attention when
```
Raw continuation:
```text
 the "I's.
```
Warnings: `none`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 23. en_ordinary_037 - average

Prompt:
```text
The museum room near closing time seemed unusually quiet because
```
Raw continuation:
```text
 of the first time.
```
Warnings: `none`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 24. en_ordinary_039 - average

Prompt:
```text
A familiar song from another room seemed unusually quiet because
```
Raw continuation:
```text
 of the same as a new function.
```
Warnings: `none`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 25. en_ordinary_040 - average

Prompt:
```text
A familiar song from another room first drew attention when
```
Raw continuation:
```text
 the "C.
```
Warnings: `none`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 26. en_ordinary_018 - weak

Prompt:
```text
While walking beside the river first drew attention when
```
Raw continuation:
```text
 the "C" is a "C" and "C" and "C" and "C" and "C" and "C" and "C
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.2188`; EOS: `False`.

### 27. en_ordinary_030 - weak

Prompt:
```text
A notebook left on the desk first drew attention when
```
Raw continuation:
```text
 the "C" is a "C" and "C" and "C" and "C" and "C" and "C" and "C
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.2188`; EOS: `False`.

### 28. en_ordinary_036 - weak

Prompt:
```text
A traveler waiting at the platform first drew attention when
```
Raw continuation:
```text
 the "C" is a "C" and "C" is a "C" and "C" and "C" and "C" and "
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.2188`; EOS: `False`.

### 29. en_ordinary_048 - weak

Prompt:
```text
A quiet seat beside the window first drew attention when
```
Raw continuation:
```text
 the "C" is a "C" and "C" and "C" and "C" and "C" and "C" and "C
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.2188`; EOS: `False`.

### 30. en_ordinary_014 - weak

Prompt:
```text
A table near the library entrance first drew attention when
```
Raw continuation:
```text
 the "Py_C_C_C_C_C_C_C_C_C_C_C_C_C_C_
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.1562`; EOS: `False`.

## Factual continuations

### 31. en_factual_003 - relative stronger

Prompt:
```text
The Earth completes one orbit around the Sun in
```
Raw continuation:
```text
 the first-in.
```
Warnings: `factual_unreliability`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 32. en_factual_006 - relative stronger

Prompt:
```text
The chemical symbol for iron is
```
Raw continuation:
```text
 the "C.
```
Warnings: `factual_unreliability`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 33. tr_factual_002 - average

Prompt:
```text
Karadeniz Türkiye'nin kuzeyinde
```
Raw continuation:
```text
, 19999'da, 1999'da, 1999'de 1'de 19
```
Warnings: `factual_unreliability, repetition`; exact loop: `True`; unique-token ratio: `0.2188`; EOS: `False`.

### 34. tr_factual_003 - average

Prompt:
```text
Dünya'nın doğal uydusu Ay
```
Raw continuation:
```text
'in, 1999'de, 199'de 1'de 1999'de 199
```
Warnings: `factual_unreliability, repetition`; exact loop: `True`; unique-token ratio: `0.2188`; EOS: `False`.

### 35. en_factual_020 - weak

Prompt:
```text
An earthquake is commonly recorded by
```
Raw continuation:
```text
 the "C" and "C" and "C" and "C" and "C" and "C" and "C" and "C"
```
Warnings: `factual_unreliability, repetition`; exact loop: `True`; unique-token ratio: `0.1562`; EOS: `False`.

## Technical and code continuations

### 36. code_structured_009 - relative stronger

Prompt:
```text
{
  "name": "darkmind-preview",
  "version":
```
Raw continuation:
```text
 "S.
```
Warnings: `code_generation_failure`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 37. en_technical_007 - relative stronger

Prompt:
```text
A transformer attention layer computes
```
Raw continuation:
```text
's.
```
Warnings: `none`; exact loop: `False`; unique-token ratio: `1.0000`; EOS: `True`.

### 38. en_technical_010 - average

Prompt:
```text
A model checkpoint should preserve optimizer state when
```
Raw continuation:
```text
 the "Py_C_C_C_C_C_C_C_C_C_C_C_C_C_C_
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.1562`; EOS: `False`.

### 39. tr_technical_001 - average

Prompt:
```text
Python'da bir listenin elemanlarını gezerken
```
Raw continuation:
```text
, "S" olarak "S" olarak "S" olarak "S" olarak "S" olarak "S" olarak "S" olarak "S"
```
Warnings: `repetition`; exact loop: `True`; unique-token ratio: `0.1562`; EOS: `False`.

### 40. code_structured_004 - weak

Prompt:
```text
SELECT id, name FROM users WHERE
```
Raw continuation:
```text
_C_C_C_C_C_C_C_C_C_C_C_C_C_C_C_C
```
Warnings: `code_generation_failure, repetition`; exact loop: `True`; unique-token ratio: `0.0625`; EOS: `False`.
