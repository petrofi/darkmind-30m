# Checkpoint and Tokenizer Compatibility Audit

Tokenizer: `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-30m\tokenizer\darkmind-tokenizer`
Base checkpoint: `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-30m\models\darkmind-30m-10k-step15000.pt`
Student checkpoint: `C:\Users\tarik\OneDrive\Ekler\Desktop\darkmind-30m\models\darkmind-30m-qwen-distill-pilot500-tr-en-v2.pt`

## Tokenizer File Hashes

- `merges.txt`: `c551acb4aa4630fef0df2c9ffdc90118be8bf5d100ee3cf5637d868d27475b7b`
- `vocab.json`: `af618a08577c2bd741d3b380f0b32bf3a0ac279caad313735621a17f088e9af7`

## Compatibility

- Tokenizer vocabulary size: `5422`
- Maximum tokenizer ID: `5421`
- Base checkpoint vocabulary size: `5422`
- Student checkpoint vocabulary size: `5422`
- Base embedding shape: `[5422, 512]`
- Base LM head shape: `[5422, 512]`
- Student embedding shape: `[5422, 512]`
- Student LM head shape: `[5422, 512]`
- Base input/output weights tied/equal in state dict: `{'present': True, 'allclose': True, 'max_abs_diff': 0.0}`
- Student input/output weights tied/equal in state dict: `{'present': True, 'allclose': True, 'max_abs_diff': 0.0}`
- Special token IDs: `{'<s>': 1, '<pad>': 0, '</s>': 2, '<unk>': 3, '<mask>': 4, '<|end|>': None}`
- EOS token ID: `2`
- PAD token ID: `0`
- UNK token ID: `3`
- Token IDs within base checkpoint bounds: `True`
- Token IDs within student checkpoint bounds: `True`
- Base checkpoint metadata keys: `['config', 'final_step_loss', 'final_train_loss', 'final_val_loss', 'model_state_dict', 'optimizer_state_dict', 'parameter_count', 'run_name', 'train_tokens', 'training_config', 'val_tokens', 'vocab_size']`
- Student checkpoint metadata keys: `['base_checkpoint', 'config', 'data_path', 'final_step_loss', 'instruction_count', 'model_state_dict', 'optimizer_state_dict', 'parameter_count', 'run_name', 'tokenizer_path', 'train_examples', 'train_label_tokens', 'train_loss', 'training_config', 'val_examples', 'val_label_tokens', 'val_loss']`

## Vocabulary Script Counts

- turkish_latin: `10`
- english_ascii: `1623`
- hebrew: `0`
- greek: `0`
- arabic: `0`
- cyrillic: `0`
- devanagari: `0`
- japanese_cjk: `0`
- replacement_or_malformed: `1843`

## Suspicious Vocabulary Entries

- `131`: `Â`
- `132`: `Ã`
- `133`: `Ä`
- `134`: `Å`
- `177`: `ð`
- `261`: `Ä±`
- `268`: `Ã¼`
- `278`: `lanÄ±`
- `283`: `Ã§`
- `286`: `ullanÄ±`
- `287`: `ÄŁ`
- `289`: `ÅŁ`
- `291`: `Ã¶`
- `294`: `cÄ±`
- `297`: `ullanÄ±cÄ±`
- `298`: `Ä±r`
- `302`: `KullanÄ±cÄ±`
- `318`: `Ä±n`
- `320`: `Ä±l`
- `346`: `asÄ±l`
- `347`: `ĠnasÄ±l`
- `356`: `ĠÃ¶`
- `358`: `ĠÃ§`
- `361`: `Ã¼k`
- `363`: `alÄ±`
- `365`: `ĠiÃ§`
- `369`: `ĠiÃ§in`
- `371`: `Ã¼r`
- `375`: `ĠkullanÄ±`
- `378`: `ÅŁt`
- `383`: `ĠgÃ¶`
- `384`: `Ä±nÄ±`
- `389`: `Ã¼Ã§`
- `393`: `ĠdeÄŁ`
- `401`: `ÄŁit`
- `406`: `ĠeÄŁit`
- `420`: `Ã¼Ã§Ã¼k`
- `421`: `lÄ±`
- `425`: `Ä±m`
- `427`: `ÄŁr`
- `431`: `ĠgÃ¶st`
- `434`: `ĠgÃ¶ster`
- `435`: `dÄ±r`
- `445`: `Ã¶z`
- `447`: `ĠeÄŁitim`
- `448`: `ĠÃ¶rn`
- `450`: `Ã¶n`
- `456`: `ĠkullanÄ±l`
- `461`: `Ä±lÄ±r`
- `464`: `ĠkullanÄ±lÄ±r`
- `465`: `Ä±k`
- `473`: `ĠdÃ¶n`
- `475`: `ĠÃ§alÄ±`
- `476`: `Ã¼n`
- `478`: `ĠkÃ¼Ã§Ã¼k`
- `481`: `ĠÃ¼`
- `484`: `iÅŁ`
- `486`: `malÄ±`
- `492`: `ĠiÅŁ`
- `498`: `ÅŁtir`
- `503`: `dÄ±`
- `512`: `Ä±sa`
- `513`: `Ã§e`
- `516`: `ĠÃ¶rnek`
- `524`: `ĠÃ§alÄ±ÅŁ`
- `529`: `kÃ§e`
- `546`: `Ã¼rkÃ§e`
- `548`: `ĠTÃ¼rkÃ§e`
- `558`: `ĠaÃ§`
- `561`: `larÄ±`
- `562`: `ĠaÃ§Ä±k`
- `568`: `ĠÃ¶ÄŁr`
- `575`: `Ã¼m`
- `576`: `ÄŁi`
- `578`: `Ä±nda`
- `579`: `ĠdoÄŁr`
- `580`: `ĠÃ¶ÄŁren`
- `590`: `asÄ±`
- `595`: `ĠdeÄŁer`
- `596`: `dÄ±ÄŁ`
- `598`: `ĠÃ§Ã¶z`
- `601`: `ĠsayÄ±`
- `603`: `atÄ±r`
- `606`: `lanÄ±r`
- `607`: `ĠsatÄ±r`
- `609`: `ĠbaÅŁ`
- `610`: `Ã¶zl`
- `611`: `ĠdoÄŁru`
- `615`: `duÄŁ`
- `616`: `ĠgeliÅŁtir`
- `620`: `ĠÃ¶n`
- `623`: `gÃ¼`
- `630`: `ĠgÃ¶sterir`
- `632`: `arÄ±`
- `633`: `ĠkÄ±sa`
- `639`: `asÄ±nÄ±`
- `653`: `ĠyazÄ±lÄ±r`
- `663`: `ĠolduÄŁ`
- `664`: `ĠkullanÄ±labilir`
- `672`: `Ã¼ÅŁ`

## Exact Round-Trip Tests

### `Merhaba, sen kimsin?`
- Token IDs: `[675, 16, 2624, 2184, 35]`
- Token strings: `['Merhaba', ',', 'Ġsen', 'Ġkimsin', '?']`
- Decoded: `Merhaba, sen kimsin?`
- Exact round-trip match: `True`
- Tokens per character: `0.25`
- Suspicious-script tokens: `[]`

### `Python kullanarak küçük bir REST servisini nasıl başlatırım?`
- Token IDs: `[1900, 2505, 478, 333, 1787, 41, 55, 56, 293, 265, 90, 77, 635, 347, 609, 4034, 425, 35]`
- Token strings: `['Python', 'Ġkullanarak', 'ĠkÃ¼Ã§Ã¼k', 'Ġbir', 'ĠR', 'E', 'S', 'T', 'Ġs', 'er', 'v', 'i', 'sini', 'ĠnasÄ±l', 'ĠbaÅŁ', 'latÄ±r', 'Ä±m', '?']`
- Decoded: `Python kullanarak küçük bir REST servisini nasıl başlatırım?`
- Exact round-trip match: `True`
- Tokens per character: `0.3`
- Suspicious-script tokens: `['ĠkÃ¼Ã§Ã¼k', 'ĠnasÄ±l', 'ĠbaÅŁ', 'latÄ±r', 'Ä±m']`

### `Docker konteynerim hemen kapanıyor.`
- Token IDs: `[40, 1128, 79, 265, 423, 351, 93, 82, 3467, 1575, 279, 3416, 263, 4084, 18]`
- Token strings: `['D', 'oc', 'k', 'er', 'Ġkon', 'te', 'y', 'n', 'erim', 'Ġhem', 'en', 'Ġkap', 'an', 'Ä±yor', '.']`
- Decoded: `Docker konteynerim hemen kapanıyor.`
- Exact round-trip match: `True`
- Tokens per character: `0.4286`
- Suspicious-script tokens: `['Ä±yor']`

### `Validation loss neden yükselir?`
- Token IDs: `[58, 1122, 713, 647, 1755, 859, 35]`
- Token strings: `['V', 'alidation', 'Ġloss', 'Ġneden', 'ĠyÃ¼ks', 'elir', '?']`
- Decoded: `Validation loss neden yükselir?`
- Exact round-trip match: `True`
- Tokens per character: `0.2258`
- Suspicious-script tokens: `['ĠyÃ¼ks']`

### `Hello, how do I create a REST API?`
- Token IDs: `[44, 353, 563, 16, 334, 2841, 557, 753, 357, 325, 876, 339, 1787, 41, 55, 56, 995, 52, 45, 35]`
- Token strings: `['H', 'el', 'lo', ',', 'Ġh', 'ow', 'Ġdo', 'ĠI', 'Ġc', 're', 'ate', 'Ġa', 'ĠR', 'E', 'S', 'T', 'ĠA', 'P', 'I', '?']`
- Decoded: `Hello, how do I create a REST API?`
- Exact round-trip match: `True`
- Tokens per character: `0.5882`
- Suspicious-script tokens: `[]`

### `A short Python function returns a list.`
- Token IDs: `[37, 293, 76, 501, 319, 348, 337, 3025, 460, 87, 339, 440, 18]`
- Token strings: `['A', 'Ġs', 'h', 'ort', 'ĠPython', 'Ġf', 'un', 'ction', 'Ġreturn', 's', 'Ġa', 'Ġlist', '.']`
- Decoded: `A short Python function returns a list.`
- Exact round-trip match: `True`
- Tokens per character: `0.3333`
- Suspicious-script tokens: `[]`


## Deterministic Training Round Trips

### Training text 1
- Original: `Explain how LoRA (Low-Rank Adaptation) reduces memory usage during fine-tuning of large language models, and why it's effective for parameter-efficient tuning.`
- Token IDs: `[41, 92, 1780, 264, 334, 2841, 940, 83, 54, 37, 1705, 48, 2841, 17, 54, 263, 79, 995, 368, 570, 612, 13, 670, 577, 379, 87, 386, 81, 2078, 754, 388, 483, 307, 345, 505, 4257, 17, 5292, 2931, 225, 349, 483, 225, 1565, 372, 87, 16, 3539, 2871, 93, 315, 88, 11, 87, 324, 4001, 507, 88, 1846, 510, 3772, 17, 73, 4001, 2431, 2074, 313, 337, 505, 18]`
- Token strings: `['E', 'x', 'pla', 'in', 'Ġh', 'ow', 'ĠL', 'o', 'R', 'A', 'Ġ(', 'L', 'ow', '-', 'R', 'an', 'k', 'ĠA', 'da', 'pt', 'ation', ')', 'Ġre', 'du', 'ce', 's', 'Ġme', 'm', 'ory', 'Ġu', 'sa', 'ge', 'Ġd', 'ur', 'ing', 'Ġfine', '-', 'tuning', 'Ġof', 'Ġ', 'lar', 'ge', 'Ġ', 'language', 'Ġmodel', 's', ',', 'Ġand', 'Ġwh', 'y', 'Ġi', 't', "'", 's', 'Ġe', 'ff', 'ec', 't', 'ive', 'Ġfor', 'Ġparameter', '-', 'e', 'ff', 'ici', 'ent', 'Ġt', 'un', 'ing', '.']`
- Decoded: `Explain how LoRA (Low-Rank Adaptation) reduces memory usage during fine-tuning of large language models, and why it's effective for parameter-efficient tuning.`
- Exact round-trip match: `True`
- Tokens per character: `0.4403`
- Suspicious-script tokens: `[]`

### Training text 2
- Original: `LoRA reduces memory usage by freezing the original model weights and only updating low-rank matrices that are multiplied with the input or output. This allows fine-tuning with significantly fewer parameters, making it effective for adapting large models on lim`
- Token IDs: `[48, 83, 54, 37, 670, 577, 379, 87, 386, 81, 2078, 754, 388, 483, 285, 93, 348, 325, 3996, 505, 3383, 73, 2869, 771, 3395, 372, 225, 2861, 87, 3539, 3386, 225, 637, 72, 311, 505, 2870, 91, 17, 490, 79, 1800, 86, 1560, 87, 3843, 2497, 4219, 469, 295, 650, 225, 91, 4016, 3383, 73, 1067, 2869, 2932, 992, 18, 382, 4014, 710, 563, 91, 87, 4257, 17, 5292, 225, 91, 4016, 818, 75, 82, 555, 77, 71, 2071, 1368, 348, 73, 890, 3772, 87, 16, 1342, 505, 315, 88, 324, 4001, 507, 88, 1846, 510, 339, 368, 570, 505, 225, 349, 483, 372, 87, 2600, 437, 2434, 650, 334, 275, 72, 2860, 73, 1108, 472, 952, 265, 90]...`
- Token strings: `['L', 'o', 'R', 'A', 'Ġre', 'du', 'ce', 's', 'Ġme', 'm', 'ory', 'Ġu', 'sa', 'ge', 'Ġb', 'y', 'Ġf', 're', 'ez', 'ing', 'Ġth', 'e', 'Ġor', 'ig', 'inal', 'Ġmodel', 'Ġ', 'weight', 's', 'Ġand', 'Ġonly', 'Ġ', 'up', 'd', 'at', 'ing', 'Ġlo', 'w', '-', 'ran', 'k', 'Ġmat', 'r', 'ice', 's', 'Ġthat', 'Ġare', 'Ġmult', 'ip', 'li', 'ed', 'Ġ', 'w', 'ith', 'Ġth', 'e', 'Ġinput', 'Ġor', 'Ġout', 'put', '.', 'ĠT', 'his', 'Ġal', 'lo', 'w', 's', 'Ġfine', '-', 'tuning', 'Ġ', 'w', 'ith', 'Ġsi', 'g', 'n', 'if', 'i', 'c', 'ant', 'ly', 'Ġf', 'e', 'wer', 'Ġparameter', 's', ',', 'Ġmak', 'ing', 'Ġi', 't', 'Ġe', 'ff', 'ec', 't', 'ive', 'Ġfor', 'Ġa', 'da', 'pt', 'ing', 'Ġ', 'lar', 'ge', 'Ġmodel', 's', 'Ġon', 'Ġli', 'mit', 'ed', 'Ġh', 'ar', 'd', 'war', 'e', 'Ġwhile', 'Ġpr', 'es', 'er', 'v']...`
- Decoded: `LoRA reduces memory usage by freezing the original model weights and only updating low-rank matrices that are multiplied with the input or output. This allows fine-tuning with significantly fewer parameters, making it effective for adapting large models on lim`
- Exact round-trip match: `True`
- Tokens per character: `0.4158`
- Suspicious-script tokens: `[]`

### Training text 3
- Original: `How do I implement a RESTful API endpoint in C# .NET to create a new user with validation, using a controller, service, and repository pattern?`
- Token IDs: `[44, 2841, 557, 753, 315, 81, 84, 468, 2074, 339, 1787, 41, 55, 56, 3347, 995, 52, 45, 646, 72, 614, 412, 409, 7, 2229, 50, 41, 56, 3446, 357, 325, 876, 339, 373, 91, 225, 1090, 225, 91, 4016, 1380, 16, 754, 387, 75, 339, 357, 266, 494, 80, 504, 16, 293, 265, 90, 1560, 16, 3539, 2792, 380, 311, 1918, 82, 35]`
- Token strings: `['H', 'ow', 'Ġdo', 'ĠI', 'Ġi', 'm', 'p', 'lem', 'ent', 'Ġa', 'ĠR', 'E', 'S', 'T', 'ful', 'ĠA', 'P', 'I', 'Ġen', 'd', 'point', 'Ġin', 'ĠC', '#', 'Ġ.', 'N', 'E', 'T', 'Ġto', 'Ġc', 're', 'ate', 'Ġa', 'Ġne', 'w', 'Ġ', 'user', 'Ġ', 'w', 'ith', 'Ġvalidation', ',', 'Ġu', 'sin', 'g', 'Ġa', 'Ġc', 'on', 'tro', 'l', 'ler', ',', 'Ġs', 'er', 'v', 'ice', ',', 'Ġand', 'Ġrepository', 'Ġp', 'at', 'ter', 'n', '?']`
- Decoded: `How do I implement a RESTful API endpoint in C# .NET to create a new user with validation, using a controller, service, and repository pattern?`
- Exact round-trip match: `True`
- Tokens per character: `0.4476`
- Suspicious-script tokens: `[]`

### Training text 4
- Original: `Create a POST endpoint in a controller that accepts a UserCreateDto, validates it, and calls a UserService. The service validates the data and calls the repository to save it to the database using Entity Framework. Return 201 Created on success, 400 Bad Reques`
- Token IDs: `[39, 325, 876, 339, 309, 51, 55, 56, 646, 72, 614, 412, 339, 357, 266, 494, 80, 504, 3843, 339, 71, 379, 570, 87, 339, 1198, 39, 325, 876, 40, 88, 83, 16, 299, 782, 876, 87, 315, 88, 16, 3539, 357, 306, 80, 87, 339, 1198, 55, 265, 90, 1560, 18, 382, 76, 73, 293, 265, 90, 1560, 299, 782, 876, 87, 3383, 73, 867, 3539, 357, 306, 80, 87, 3383, 73, 2792, 3446, 572, 1054, 315, 88, 3446, 3383, 73, 867, 2422, 754, 387, 75, 1056, 82, 88, 1026, 721, 86, 367, 91, 290, 79, 18, 1787, 390, 459, 1535, 21, 409, 325, 311, 650, 2600, 293, 1232, 379, 87, 87, 16, 794, 1764, 399, 593, 1787, 73]...`
- Token strings: `['C', 're', 'ate', 'Ġa', 'ĠP', 'O', 'S', 'T', 'Ġen', 'd', 'point', 'Ġin', 'Ġa', 'Ġc', 'on', 'tro', 'l', 'ler', 'Ġthat', 'Ġa', 'c', 'ce', 'pt', 's', 'Ġa', 'ĠUser', 'C', 're', 'ate', 'D', 't', 'o', ',', 'Ġv', 'alid', 'ate', 's', 'Ġi', 't', ',', 'Ġand', 'Ġc', 'al', 'l', 's', 'Ġa', 'ĠUser', 'S', 'er', 'v', 'ice', '.', 'ĠT', 'h', 'e', 'Ġs', 'er', 'v', 'ice', 'Ġv', 'alid', 'ate', 's', 'Ġth', 'e', 'Ġdata', 'Ġand', 'Ġc', 'al', 'l', 's', 'Ġth', 'e', 'Ġrepository', 'Ġto', 'Ġsa', 've', 'Ġi', 't', 'Ġto', 'Ġth', 'e', 'Ġdata', 'base', 'Ġu', 'sin', 'g', 'ĠE', 'n', 't', 'ity', 'ĠF', 'r', 'ame', 'w', 'or', 'k', '.', 'ĠR', 'et', 'urn', 'Ġ20', '1', 'ĠC', 're', 'at', 'ed', 'Ġon', 'Ġs', 'uc', 'ce', 's', 's', ',', 'Ġ4', '00', 'ĠB', 'ad', 'ĠR', 'e']...`
- Decoded: `Create a POST endpoint in a controller that accepts a UserCreateDto, validates it, and calls a UserService. The service validates the data and calls the repository to save it to the database using Entity Framework. Return 201 Created on success, 400 Bad Reques`
- Exact round-trip match: `True`
- Tokens per character: `0.4577`
- Suspicious-script tokens: `[]`

### Training text 5
- Original: `A farmer has 12 apples and wants to distribute them equally among 4 children. How many apples does each child get, and how many are left over?`
- Token IDs: `[37, 348, 275, 81, 265, 2937, 3533, 1485, 84, 2633, 3539, 225, 91, 2071, 87, 3446, 1935, 733, 2319, 3383, 392, 324, 1106, 306, 1368, 339, 81, 2238, 794, 357, 76, 1794, 86, 279, 18, 802, 2841, 2085, 93, 1485, 84, 2633, 557, 952, 324, 69, 708, 357, 76, 1794, 1045, 16, 3539, 334, 2841, 2085, 93, 2497, 225, 321, 74, 88, 329, 709, 35]`
- Token strings: `['A', 'Ġf', 'ar', 'm', 'er', 'Ġhas', 'Ġ12', 'Ġap', 'p', 'les', 'Ġand', 'Ġ', 'w', 'ant', 's', 'Ġto', 'Ġdi', 'str', 'ibute', 'Ġth', 'em', 'Ġe', 'qu', 'al', 'ly', 'Ġa', 'm', 'ong', 'Ġ4', 'Ġc', 'h', 'ild', 'r', 'en', '.', 'ĠH', 'ow', 'Ġman', 'y', 'Ġap', 'p', 'les', 'Ġdo', 'es', 'Ġe', 'a', 'ch', 'Ġc', 'h', 'ild', 'Ġget', ',', 'Ġand', 'Ġh', 'ow', 'Ġman', 'y', 'Ġare', 'Ġ', 'le', 'f', 't', 'Ġo', 'ver', '?']`
- Decoded: `A farmer has 12 apples and wants to distribute them equally among 4 children. How many apples does each child get, and how many are left over?`
- Exact round-trip match: `True`
- Tokens per character: `0.4577`
- Suspicious-script tokens: `[]`

### Training text 6
- Original: `Each child gets 3 apples, and there are 0 apples left over. This is calculated by dividing 12 by 4, which equals 3 with no remainder.`
- Token IDs: `[41, 69, 708, 357, 76, 1794, 1045, 87, 542, 1485, 84, 2633, 16, 3539, 3383, 2237, 2497, 541, 1485, 84, 2633, 225, 321, 74, 88, 329, 709, 18, 382, 4014, 894, 357, 306, 71, 271, 311, 650, 285, 93, 1203, 77, 1513, 3533, 285, 93, 794, 16, 2871, 77, 708, 324, 1106, 306, 87, 542, 225, 91, 4016, 2904, 670, 328, 3520, 18]`
- Token strings: `['E', 'a', 'ch', 'Ġc', 'h', 'ild', 'Ġget', 's', 'Ġ3', 'Ġap', 'p', 'les', ',', 'Ġand', 'Ġth', 'ere', 'Ġare', 'Ġ0', 'Ġap', 'p', 'les', 'Ġ', 'le', 'f', 't', 'Ġo', 'ver', '.', 'ĠT', 'his', 'Ġis', 'Ġc', 'al', 'c', 'ul', 'at', 'ed', 'Ġb', 'y', 'Ġdiv', 'i', 'ding', 'Ġ12', 'Ġb', 'y', 'Ġ4', ',', 'Ġwh', 'i', 'ch', 'Ġe', 'qu', 'al', 's', 'Ġ3', 'Ġ', 'w', 'ith', 'Ġno', 'Ġre', 'ma', 'inder', '.']`
- Decoded: `Each child gets 3 apples, and there are 0 apples left over. This is calculated by dividing 12 by 4, which equals 3 with no remainder.`
- Exact round-trip match: `True`
- Tokens per character: `0.4737`
- Suspicious-script tokens: `[]`

### Training text 7
- Original: `C# .NET Web API'de bir kullanıcı kaydı yapmak için gerekli olan controller, service ve repository yapılarını nasıl oluştururum?`
- Token IDs: `[39, 7, 2229, 50, 41, 56, 2599, 3995, 995, 52, 45, 322, 73, 333, 996, 907, 503, 2477, 369, 3619, 1574, 357, 266, 494, 80, 504, 16, 293, 265, 90, 1560, 338, 2792, 394, 2084, 384, 347, 887, 1027, 35]`
- Token strings: `['C', '#', 'Ġ.', 'N', 'E', 'T', 'ĠW', 'eb', 'ĠA', 'P', 'I', "'d", 'e', 'Ġbir', 'ĠkullanÄ±cÄ±', 'Ġkay', 'dÄ±', 'Ġyapmak', 'ĠiÃ§in', 'Ġgerekli', 'Ġolan', 'Ġc', 'on', 'tro', 'l', 'ler', ',', 'Ġs', 'er', 'v', 'ice', 'Ġve', 'Ġrepository', 'Ġyap', 'Ä±lar', 'Ä±nÄ±', 'ĠnasÄ±l', 'ĠoluÅŁtur', 'urum', '?']`
- Decoded: `C# .NET Web API'de bir kullanıcı kaydı yapmak için gerekli olan controller, service ve repository yapılarını nasıl oluştururum?`
- Exact round-trip match: `True`
- Tokens per character: `0.315`
- Suspicious-script tokens: `['ĠkullanÄ±cÄ±', 'dÄ±', 'ĠiÃ§in', 'Ä±lar', 'Ä±nÄ±', 'ĠnasÄ±l', 'ĠoluÅŁtur']`

### Training text 8
- Original: `Controller, kullanıcı verilerini almak için POST metodunu kullanır. Service, veriyi doğrulayıp repository'ye gönderir. Repository, Entity Framework ile veritabanına kaydeder. DTO kullanarak veri aktarımı güvenli ve temiz olur.`
- Token IDs: `[39, 266, 494, 80, 504, 16, 996, 1010, 4327, 1402, 369, 309, 51, 55, 56, 2284, 585, 2476, 18, 509, 265, 90, 1560, 16, 1144, 579, 271, 1343, 84, 2792, 11, 588, 2655, 269, 18, 1787, 73, 1979, 2078, 16, 1056, 82, 88, 1026, 721, 86, 367, 91, 290, 79, 432, 482, 335, 518, 263, 1047, 3713, 18, 428, 56, 51, 2505, 410, 339, 79, 957, 4334, 886, 338, 985, 759, 18]`
- Token strings: `['C', 'on', 'tro', 'l', 'ler', ',', 'ĠkullanÄ±cÄ±', 'Ġveril', 'erini', 'Ġalmak', 'ĠiÃ§in', 'ĠP', 'O', 'S', 'T', 'Ġmetod', 'unu', 'ĠkullanÄ±r', '.', 'ĠS', 'er', 'v', 'ice', ',', 'Ġveriyi', 'ĠdoÄŁr', 'ul', 'ayÄ±', 'p', 'Ġrepository', "'", 'ye', 'ĠgÃ¶nder', 'ir', '.', 'ĠR', 'e', 'posit', 'ory', ',', 'ĠE', 'n', 't', 'ity', 'ĠF', 'r', 'ame', 'w', 'or', 'k', 'Ġile', 'Ġver', 'it', 'ab', 'an', 'Ä±na', 'Ġkaydeder', '.', 'ĠD', 'T', 'O', 'Ġkullanarak', 'Ġveri', 'Ġa', 'k', 'tar', 'Ä±mÄ±', 'ĠgÃ¼venli', 'Ġve', 'Ġtemiz', 'Ġolur', '.']`
- Decoded: `Controller, kullanıcı verilerini almak için POST metodunu kullanır. Service, veriyi doğrulayıp repository'ye gönderir. Repository, Entity Framework ile veritabanına kaydeder. DTO kullanarak veri aktarımı güvenli ve temiz olur.`
- Exact round-trip match: `True`
- Tokens per character: `0.3186`
- Suspicious-script tokens: `['ĠkullanÄ±cÄ±', 'ĠiÃ§in', 'ĠkullanÄ±r', 'ĠdoÄŁr', 'ayÄ±', 'ĠgÃ¶nder', 'Ä±na', 'Ä±mÄ±', 'ĠgÃ¼venli']`

### Training text 9
- Original: `Bir proje için çok fazla zaman harcadım ama hâlâ istediğim sonucu alamadım. Bu durum beni yoruyor ve motive olamıyorum.`
- Token IDs: `[1104, 553, 369, 1029, 1073, 1737, 334, 275, 71, 593, 425, 728, 2051, 2262, 73, 1348, 344, 2384, 710, 2060, 1193, 18, 530, 1044, 285, 522, 305, 290, 89, 605, 338, 323, 624, 1846, 329, 591, 261, 1011, 18]`
- Token strings: `['Bir', 'Ġproje', 'ĠiÃ§in', 'ĠÃ§ok', 'Ġfazla', 'Ġzaman', 'Ġh', 'ar', 'c', 'ad', 'Ä±m', 'Ġama', 'ĠhÃ¢lÃ¢', 'Ġist', 'e', 'diÄŁ', 'im', 'Ġsonucu', 'Ġal', 'ama', 'dÄ±m', '.', 'ĠBu', 'Ġdurum', 'Ġb', 'eni', 'Ġy', 'or', 'u', 'yor', 'Ġve', 'Ġm', 'ot', 'ive', 'Ġo', 'lam', 'Ä±', 'yorum', '.']`
- Decoded: `Bir proje için çok fazla zaman harcadım ama hâlâ istediğim sonucu alamadım. Bu durum beni yoruyor ve motive olamıyorum.`
- Exact round-trip match: `True`
- Tokens per character: `0.3277`
- Suspicious-script tokens: `['ĠiÃ§in', 'ĠÃ§ok', 'Ä±m', 'ĠhÃ¢lÃ¢', 'diÄŁ', 'dÄ±m', 'Ä±']`

### Training text 10
- Original: `Bu durum gerçekten zor olabilir, özellikle çok çaba harcadıktan sonra beklenen sonuç gelmediğinde. Küçük bir adım atmayı dene: bugün sadece 10 dakika kodunuzu gözden geçirip bir hatayı düzeltmeye çalış.`
- Token IDs: `[769, 1044, 2533, 2500, 868, 16, 2020, 1029, 358, 518, 69, 334, 275, 71, 593, 465, 1917, 1031, 1408, 2132, 2257, 280, 1348, 1788, 18, 689, 333, 1540, 853, 1481, 3635, 30, 959, 2826, 1118, 1179, 3437, 699, 69, 1082, 585, 94, 89, 383, 94, 403, 1358, 269, 469, 333, 1650, 1938, 353, 88, 4118, 524, 18]`
- Token strings: `['Bu', 'Ġdurum', 'ĠgerÃ§ekten', 'Ġzor', 'Ġolabilir', ',', 'ĠÃ¶zellikle', 'ĠÃ§ok', 'ĠÃ§', 'ab', 'a', 'Ġh', 'ar', 'c', 'ad', 'Ä±k', 'tan', 'Ġsonra', 'Ġbeklenen', 'ĠsonuÃ§', 'Ġgel', 'me', 'diÄŁ', 'inde', '.', 'ĠKÃ¼Ã§Ã¼k', 'Ġbir', 'ĠadÄ±m', 'Ġat', 'mayÄ±', 'Ġdene', ':', 'Ġbu', 'gÃ¼n', 'Ġsadece', 'Ġ10', 'Ġdak', 'ik', 'a', 'Ġkod', 'unu', 'z', 'u', 'ĠgÃ¶', 'z', 'den', 'ĠgeÃ§', 'ir', 'ip', 'Ġbir', 'ĠhatayÄ±', 'ĠdÃ¼z', 'el', 't', 'meye', 'ĠÃ§alÄ±ÅŁ', '.']`
- Decoded: `Bu durum gerçekten zor olabilir, özellikle çok çaba harcadıktan sonra beklenen sonuç gelmediğinde. Küçük bir adım atmayı dene: bugün sadece 10 dakika kodunuzu gözden geçirip bir hatayı düzeltmeye çalış.`
- Exact round-trip match: `True`
- Tokens per character: `0.2822`
- Suspicious-script tokens: `['ĠgerÃ§ekten', 'ĠÃ¶zellikle', 'ĠÃ§ok', 'ĠÃ§', 'Ä±k', 'ĠsonuÃ§', 'diÄŁ', 'ĠKÃ¼Ã§Ã¼k', 'ĠadÄ±m', 'mayÄ±', 'gÃ¼n', 'ĠgÃ¶', 'ĠgeÃ§', 'ĠhatayÄ±', 'ĠdÃ¼z', 'ĠÃ§alÄ±ÅŁ']`

### Training text 11
- Original: `Python'da bir sayı dizisindeki en büyük sayıyı bulan bir kod yaz.`
- Token IDs: `[1900, 322, 69, 333, 601, 1747, 2656, 646, 1218, 2314, 722, 263, 333, 1082, 374, 18]`
- Token strings: `['Python', "'d", 'a', 'Ġbir', 'ĠsayÄ±', 'Ġdizi', 'sindeki', 'Ġen', 'ĠbÃ¼yÃ¼k', 'ĠsayÄ±yÄ±', 'Ġbul', 'an', 'Ġbir', 'Ġkod', 'Ġyaz', '.']`
- Decoded: `Python'da bir sayı dizisindeki en büyük sayıyı bulan bir kod yaz.`
- Exact round-trip match: `True`
- Tokens per character: `0.2462`
- Suspicious-script tokens: `['ĠsayÄ±', 'ĠbÃ¼yÃ¼k', 'ĠsayÄ±yÄ±']`

### Training text 12
- Original: `Bir sayı dizisindeki en büyük sayıyı bulmak için `max()` fonksiyonunu kullanabilirsin. Örneğin: `sayilar = [10, 5, 8, 20, 3]; en_buyuk = max(sayilar)` komutu ile en büyük sayıyı elde edersin. Bu kod, dizideki tüm sayıları kontrol eder ve en büyüğünü döndürür.`
- Token IDs: `[1104, 601, 1747, 2656, 646, 1218, 2314, 722, 474, 369, 225, 68, 2928, 527, 68, 626, 585, 702, 1180, 18, 3109, 5166, 30, 225, 68, 87, 355, 288, 275, 308, 417, 1456, 16, 644, 16, 2598, 16, 1535, 16, 542, 65, 31, 646, 67, 70, 1164, 1784, 308, 4222, 12, 87, 355, 288, 275, 13, 68, 4843, 432, 646, 1218, 2314, 324, 80, 270, 1359, 387, 18, 530, 1082, 16, 4865, 313, 575, 1356, 497, 1359, 338, 646, 908, 1081, 1146, 1508, 18]`
- Token strings: `['Bir', 'ĠsayÄ±', 'Ġdizi', 'sindeki', 'Ġen', 'ĠbÃ¼yÃ¼k', 'ĠsayÄ±yÄ±', 'Ġbul', 'mak', 'ĠiÃ§in', 'Ġ', '`', 'max', '()', '`', 'Ġfonksiyon', 'unu', 'Ġkullan', 'abilirsin', '.', 'ĠÃĸrn', 'eÄŁin', ':', 'Ġ', '`', 's', 'ay', 'il', 'ar', 'Ġ=', 'Ġ[', '10', ',', 'Ġ5', ',', 'Ġ8', ',', 'Ġ20', ',', 'Ġ3', ']', ';', 'Ġen', '_', 'b', 'uy', 'uk', 'Ġ=', 'Ġmax', '(', 's', 'ay', 'il', 'ar', ')', '`', 'Ġkomutu', 'Ġile', 'Ġen', 'ĠbÃ¼yÃ¼k', 'ĠsayÄ±yÄ±', 'Ġe', 'l', 'de', 'Ġeder', 'sin', '.', 'ĠBu', 'Ġkod', ',', 'Ġdizideki', 'Ġt', 'Ã¼m', 'ĠsayÄ±larÄ±', 'Ġkontrol', 'Ġeder', 'Ġve', 'Ġen', 'ĠbÃ¼y', 'Ã¼ÄŁ', 'Ã¼nÃ¼', 'ĠdÃ¶ndÃ¼rÃ¼r', '.']`
- Decoded: `Bir sayı dizisindeki en büyük sayıyı bulmak için `max()` fonksiyonunu kullanabilirsin. Örneğin: `sayilar = [10, 5, 8, 20, 3]; en_buyuk = max(sayilar)` komutu ile en büyük sayıyı elde edersin. Bu kod, dizideki tüm sayıları kontrol eder ve en büyüğünü döndürür.`
- Exact round-trip match: `True`
- Tokens per character: `0.3205`
- Suspicious-script tokens: `['ĠsayÄ±', 'ĠbÃ¼yÃ¼k', 'ĠsayÄ±yÄ±', 'ĠiÃ§in', 'ĠÃĸrn', 'eÄŁin', 'ĠbÃ¼yÃ¼k', 'ĠsayÄ±yÄ±', 'Ã¼m', 'ĠsayÄ±larÄ±', 'ĠbÃ¼y', 'Ã¼ÄŁ', 'Ã¼nÃ¼', 'ĠdÃ¶ndÃ¼rÃ¼r']`

### Training text 13
- Original: `What's the best way to loop through a list in Python and access both index and value?`
- Token IDs: `[59, 1162, 11, 87, 3383, 73, 285, 73, 273, 225, 91, 355, 3446, 3806, 3383, 404, 89, 75, 76, 339, 440, 412, 319, 3539, 339, 71, 379, 87, 87, 874, 277, 1395, 3539, 1084, 35]`
- Token strings: `['W', 'hat', "'", 's', 'Ġth', 'e', 'Ġb', 'e', 'st', 'Ġ', 'w', 'ay', 'Ġto', 'Ġloop', 'Ġth', 'ro', 'u', 'g', 'h', 'Ġa', 'Ġlist', 'Ġin', 'ĠPython', 'Ġand', 'Ġa', 'c', 'ce', 's', 's', 'Ġbo', 'th', 'Ġindex', 'Ġand', 'Ġvalue', '?']`
- Decoded: `What's the best way to loop through a list in Python and access both index and value?`
- Exact round-trip match: `True`
- Tokens per character: `0.4118`
- Suspicious-script tokens: `[]`

### Training text 14
- Original: `Use enumerate() to get both index and value in a for loop. It’s cleaner than using range(len()) and avoids index errors. Example: for i, item in enumerate(my_list): print(i, item).`
- Token IDs: `[57, 430, 1097, 527, 3446, 1045, 874, 277, 1395, 3539, 1084, 412, 339, 510, 3806, 18, 753, 88, 163, 227, 252, 87, 4272, 265, 3383, 263, 754, 387, 75, 918, 12, 471, 913, 3539, 339, 90, 83, 1698, 87, 1395, 1707, 87, 18, 1056, 92, 2306, 321, 30, 510, 315, 16, 1478, 412, 1097, 12, 3358, 67, 3432, 424, 457, 12, 77, 16, 1478, 3902]`
- Token strings: `['U', 'se', 'Ġenumerate', '()', 'Ġto', 'Ġget', 'Ġbo', 'th', 'Ġindex', 'Ġand', 'Ġvalue', 'Ġin', 'Ġa', 'Ġfor', 'Ġloop', '.', 'ĠI', 't', 'â', 'Ģ', 'Ļ', 's', 'Ġclean', 'er', 'Ġth', 'an', 'Ġu', 'sin', 'g', 'Ġrange', '(', 'len', '())', 'Ġand', 'Ġa', 'v', 'o', 'id', 's', 'Ġindex', 'Ġerror', 's', '.', 'ĠE', 'x', 'amp', 'le', ':', 'Ġfor', 'Ġi', ',', 'Ġitem', 'Ġin', 'Ġenumerate', '(', 'my', '_', 'list', '):', 'Ġprint', '(', 'i', ',', 'Ġitem', ').']`
- Decoded: `Use enumerate() to get both index and value in a for loop. It’s cleaner than using range(len()) and avoids index errors. Example: for i, item in enumerate(my_list): print(i, item).`
- Exact round-trip match: `True`
- Tokens per character: `0.3611`
- Suspicious-script tokens: `[]`

### Training text 15
- Original: `Python projesinde 'ModuleNotFoundError: No module named 'tensorflow'' hatası alıyorum. Sanal ortamda tensorflow kurulu ama hata devam ediyor. Ne yapmalıyım?`
- Token IDs: `[1900, 1651, 1079, 49, 2277, 1688, 30, 1984, 2924, 3037, 1079, 625, 955, 74, 563, 91, 11, 11, 869, 1186, 18, 3009, 2715, 2917, 74, 563, 91, 2008, 728, 519, 1533, 4226, 605, 18, 1150, 1112, 35]`
- Token strings: `['Python', 'Ġprojesinde', "Ġ'", 'M', 'odule', 'NotFoundError', ':', 'ĠNo', 'Ġmodule', 'Ġnamed', "Ġ'", 'ten', 'sor', 'f', 'lo', 'w', "'", "'", 'ĠhatasÄ±', 'ĠalÄ±yorum', '.', 'ĠSanal', 'Ġortamda', 'Ġtensor', 'f', 'lo', 'w', 'Ġkurulu', 'Ġama', 'Ġhata', 'Ġdevam', 'Ġedi', 'yor', '.', 'ĠNe', 'ĠyapmalÄ±yÄ±m', '?']`
- Decoded: `Python projesinde 'ModuleNotFoundError: No module named 'tensorflow'' hatası alıyorum. Sanal ortamda tensorflow kurulu ama hata devam ediyor. Ne yapmalıyım?`
- Exact round-trip match: `True`
- Tokens per character: `0.2372`
- Suspicious-script tokens: `['ĠhatasÄ±', 'ĠalÄ±yorum', 'ĠyapmalÄ±yÄ±m']`

### Training text 16
- Original: `Sanal ortamın etkin olduğunu kontrol edin. Eğer etkinse, 'python -m pip install tensorflow' komutuyla tekrar yükleyin. Eğer sanal ortam farklı bir Python sürümünde oluşturulmuşsa, yeni bir sanal ortam oluşturup tekrar yükleyin.`
- Token IDs: `[55, 2873, 2714, 584, 79, 264, 1120, 497, 880, 264, 18, 3100, 584, 79, 264, 430, 16, 1079, 330, 1167, 81, 975, 412, 273, 306, 80, 2917, 74, 563, 91, 11, 1661, 1164, 316, 931, 1381, 1479, 264, 18, 3100, 1550, 917, 1185, 333, 319, 2374, 1207, 887, 271, 3362, 388, 16, 747, 333, 1550, 917, 2340, 931, 1381, 1479, 264, 18]`
- Token strings: `['S', 'anal', 'ĠortamÄ±n', 'Ġet', 'k', 'in', 'ĠolduÄŁunu', 'Ġkontrol', 'Ġed', 'in', '.', 'ĠEÄŁer', 'Ġet', 'k', 'in', 'se', ',', "Ġ'", 'python', 'Ġ-', 'm', 'Ġpip', 'Ġin', 'st', 'al', 'l', 'Ġtensor', 'f', 'lo', 'w', "'", 'Ġkomut', 'uy', 'la', 'Ġtekrar', 'ĠyÃ¼k', 'ley', 'in', '.', 'ĠEÄŁer', 'Ġsanal', 'Ġortam', 'ĠfarklÄ±', 'Ġbir', 'ĠPython', 'ĠsÃ¼rÃ¼m', 'Ã¼nde', 'ĠoluÅŁtur', 'ul', 'muÅŁ', 'sa', ',', 'Ġyeni', 'Ġbir', 'Ġsanal', 'Ġortam', 'ĠoluÅŁturup', 'Ġtekrar', 'ĠyÃ¼k', 'ley', 'in', '.']`
- Decoded: `Sanal ortamın etkin olduğunu kontrol edin. Eğer etkinse, 'python -m pip install tensorflow' komutuyla tekrar yükleyin. Eğer sanal ortam farklı bir Python sürümünde oluşturulmuşsa, yeni bir sanal ortam oluşturup tekrar yükleyin.`
- Exact round-trip match: `True`
- Tokens per character: `0.2731`
- Suspicious-script tokens: `['ĠortamÄ±n', 'ĠolduÄŁunu', 'ĠEÄŁer', 'ĠyÃ¼k', 'ĠEÄŁer', 'ĠfarklÄ±', 'ĠsÃ¼rÃ¼m', 'Ã¼nde', 'ĠoluÅŁtur', 'muÅŁ', 'ĠoluÅŁturup', 'ĠyÃ¼k']`

### Training text 17
- Original: `C#’da bir sınıf oluşturun ve içinde bir metot yazın. Bu metot, bir sayı dizisindeki en büyük sayıyı döndürsün. Dizi boşsa -1 döndürsün.`
- Token IDs: `[39, 7, 163, 227, 252, 368, 333, 1202, 887, 337, 338, 2473, 333, 4321, 374, 318, 18, 530, 4321, 16, 333, 601, 1747, 2656, 646, 1218, 2314, 1251, 87, 476, 18, 428, 385, 77, 2002, 388, 1167, 21, 1251, 87, 476, 18]`
- Token strings: `['C', '#', 'â', 'Ģ', 'Ļ', 'da', 'Ġbir', 'ĠsÄ±nÄ±f', 'ĠoluÅŁtur', 'un', 'Ġve', 'ĠiÃ§inde', 'Ġbir', 'Ġmetot', 'Ġyaz', 'Ä±n', '.', 'ĠBu', 'Ġmetot', ',', 'Ġbir', 'ĠsayÄ±', 'Ġdizi', 'sindeki', 'Ġen', 'ĠbÃ¼yÃ¼k', 'ĠsayÄ±yÄ±', 'ĠdÃ¶ndÃ¼r', 's', 'Ã¼n', '.', 'ĠD', 'iz', 'i', 'ĠboÅŁ', 'sa', 'Ġ-', '1', 'ĠdÃ¶ndÃ¼r', 's', 'Ã¼n', '.']`
- Decoded: `C#’da bir sınıf oluşturun ve içinde bir metot yazın. Bu metot, bir sayı dizisindeki en büyük sayıyı döndürsün. Dizi boşsa -1 döndürsün.`
- Exact round-trip match: `True`
- Tokens per character: `0.3111`
- Suspicious-script tokens: `['ĠsÄ±nÄ±f', 'ĠoluÅŁtur', 'ĠiÃ§inde', 'Ä±n', 'ĠsayÄ±', 'ĠbÃ¼yÃ¼k', 'ĠsayÄ±yÄ±', 'ĠdÃ¶ndÃ¼r', 'Ã¼n', 'ĠboÅŁ', 'ĠdÃ¶ndÃ¼r', 'Ã¼n']`

### Training text 18
- Original: `Sınıf adı `ArrayHelper` olsun. Metot `FindMax` adlı bir `int` döndüren yöntem olacak. Diziyi `foreach` ile dolaşarak en büyük sayıyı bulabilirsiniz. Boş dizide -1 döndürmek için `if (array.Length == 0)` kontrolü yapın.`
- Token IDs: `[3955, 1805, 225, 68, 37, 86, 86, 355, 44, 353, 556, 68, 4285, 18, 1922, 390, 624, 225, 68, 42, 416, 49, 3979, 68, 4473, 333, 225, 68, 317, 68, 3115, 2753, 329, 316, 2423, 18, 428, 385, 77, 531, 225, 68, 528, 3994, 708, 68, 432, 557, 316, 289, 1519, 646, 1218, 2314, 722, 1180, 385, 18, 4318, 1747, 270, 1167, 21, 1251, 455, 369, 225, 68, 555, 1705, 275, 86, 355, 18, 48, 279, 3348, 602, 541, 13, 68, 1437, 394, 318, 18]`
- Token strings: `['SÄ±nÄ±f', 'ĠadÄ±', 'Ġ', '`', 'A', 'r', 'r', 'ay', 'H', 'el', 'per', '`', 'Ġolsun', '.', 'ĠM', 'et', 'ot', 'Ġ', '`', 'F', 'ind', 'M', 'ax', '`', 'ĠadlÄ±', 'Ġbir', 'Ġ', '`', 'int', '`', 'ĠdÃ¶ndÃ¼ren', 'ĠyÃ¶ntem', 'Ġo', 'la', 'cak', '.', 'ĠD', 'iz', 'i', 'yi', 'Ġ', '`', 'for', 'ea', 'ch', '`', 'Ġile', 'Ġdo', 'la', 'ÅŁ', 'arak', 'Ġen', 'ĠbÃ¼yÃ¼k', 'ĠsayÄ±yÄ±', 'Ġbul', 'abilirsin', 'iz', '.', 'ĠBoÅŁ', 'Ġdizi', 'de', 'Ġ-', '1', 'ĠdÃ¶ndÃ¼r', 'mek', 'ĠiÃ§in', 'Ġ', '`', 'if', 'Ġ(', 'ar', 'r', 'ay', '.', 'L', 'en', 'gth', 'Ġ==', 'Ġ0', ')', '`', 'ĠkontrolÃ¼', 'Ġyap', 'Ä±n', '.']`
- Decoded: `Sınıf adı `ArrayHelper` olsun. Metot `FindMax` adlı bir `int` döndüren yöntem olacak. Diziyi `foreach` ile dolaşarak en büyük sayıyı bulabilirsiniz. Boş dizide -1 döndürmek için `if (array.Length == 0)` kontrolü yapın.`
- Exact round-trip match: `True`
- Tokens per character: `0.3899`
- Suspicious-script tokens: `['SÄ±nÄ±f', 'ĠadÄ±', 'ĠadlÄ±', 'ĠdÃ¶ndÃ¼ren', 'ĠyÃ¶ntem', 'ÅŁ', 'ĠbÃ¼yÃ¼k', 'ĠsayÄ±yÄ±', 'ĠBoÅŁ', 'ĠdÃ¶ndÃ¼r', 'ĠiÃ§in', 'ĠkontrolÃ¼', 'Ä±n']`

### Training text 19
- Original: `After a git push, I get 'rejected non-fast-forward' error. What’s the safest way to resolve this without losing work?`
- Token IDs: `[37, 74, 1918, 339, 1298, 2788, 16, 753, 1045, 1079, 325, 2215, 658, 296, 266, 17, 74, 69, 273, 17, 3562, 11, 1707, 18, 2599, 1162, 163, 227, 252, 87, 3383, 73, 3572, 273, 225, 91, 355, 3446, 670, 87, 4588, 3383, 597, 225, 91, 4016, 83, 453, 2870, 387, 75, 225, 91, 290, 79, 35]`
- Token strings: `['A', 'f', 'ter', 'Ġa', 'Ġgit', 'Ġpush', ',', 'ĠI', 'Ġget', "Ġ'", 're', 'jec', 'ted', 'Ġn', 'on', '-', 'f', 'a', 'st', '-', 'forward', "'", 'Ġerror', '.', 'ĠW', 'hat', 'â', 'Ģ', 'Ļ', 's', 'Ġth', 'e', 'Ġsafe', 'st', 'Ġ', 'w', 'ay', 'Ġto', 'Ġre', 's', 'olve', 'Ġth', 'is', 'Ġ', 'w', 'ith', 'o', 'ut', 'Ġlo', 'sin', 'g', 'Ġ', 'w', 'or', 'k', '?']`
- Decoded: `After a git push, I get 'rejected non-fast-forward' error. What’s the safest way to resolve this without losing work?`
- Exact round-trip match: `True`
- Tokens per character: `0.4786`
- Suspicious-script tokens: `[]`

### Training text 20
- Original: `Run `git pull --rebase` to update your local branch with remote changes, then push again. This avoids merge commits and keeps history linear. Ensure no uncommitted changes are present before pulling.`
- Token IDs: `[54, 337, 225, 68, 4009, 2965, 3108, 325, 2422, 68, 3446, 225, 637, 3338, 305, 83, 345, 5167, 2783, 225, 91, 4016, 2791, 357, 3350, 483, 87, 16, 3383, 279, 2788, 339, 75, 69, 264, 18, 382, 4014, 339, 90, 83, 1698, 87, 323, 265, 483, 2518, 87, 3539, 1469, 73, 1129, 334, 1075, 2078, 1985, 275, 18, 1056, 82, 1782, 2904, 2444, 2578, 2434, 658, 357, 3350, 483, 87, 2497, 472, 952, 2074, 3854, 2965, 505, 18]`
- Token strings: `['R', 'un', 'Ġ', '`', 'git', 'Ġpull', 'Ġ--', 're', 'base', '`', 'Ġto', 'Ġ', 'up', 'date', 'Ġy', 'o', 'ur', 'Ġlocal', 'Ġbranch', 'Ġ', 'w', 'ith', 'Ġremote', 'Ġc', 'han', 'ge', 's', ',', 'Ġth', 'en', 'Ġpush', 'Ġa', 'g', 'a', 'in', '.', 'ĠT', 'his', 'Ġa', 'v', 'o', 'id', 's', 'Ġm', 'er', 'ge', 'Ġcommit', 's', 'Ġand', 'Ġke', 'e', 'ps', 'Ġh', 'ist', 'ory', 'Ġline', 'ar', '.', 'ĠE', 'n', 'sure', 'Ġno', 'Ġun', 'com', 'mit', 'ted', 'Ġc', 'han', 'ge', 's', 'Ġare', 'Ġpr', 'es', 'ent', 'Ġbefore', 'Ġpull', 'ing', '.']`
- Decoded: `Run `git pull --rebase` to update your local branch with remote changes, then push again. This avoids merge commits and keeps history linear. Ensure no uncommitted changes are present before pulling.`
- Exact round-trip match: `True`
- Tokens per character: `0.392`
- Suspicious-script tokens: `[]`

### Training text 21
- Original: `Compare using a list versus a tuple in Python for storing a fixed set of values. What are the trade-offs?`
- Token IDs: `[39, 851, 2843, 73, 754, 387, 75, 339, 440, 482, 87, 487, 339, 1528, 412, 319, 510, 1333, 290, 505, 339, 348, 77, 92, 650, 711, 2931, 1084, 87, 18, 2599, 1162, 2497, 3383, 73, 313, 508, 270, 17, 83, 4001, 87, 35]`
- Token strings: `['C', 'om', 'par', 'e', 'Ġu', 'sin', 'g', 'Ġa', 'Ġlist', 'Ġver', 's', 'us', 'Ġa', 'Ġtuple', 'Ġin', 'ĠPython', 'Ġfor', 'Ġst', 'or', 'ing', 'Ġa', 'Ġf', 'i', 'x', 'ed', 'Ġset', 'Ġof', 'Ġvalue', 's', '.', 'ĠW', 'hat', 'Ġare', 'Ġth', 'e', 'Ġt', 'ra', 'de', '-', 'o', 'ff', 's', '?']`
- Decoded: `Compare using a list versus a tuple in Python for storing a fixed set of values. What are the trade-offs?`
- Exact round-trip match: `True`
- Tokens per character: `0.4095`
- Suspicious-script tokens: `[]`

### Training text 22
- Original: `Lists are mutable and allow modifications; tuples are immutable and faster to access. Use tuples for fixed data to ensure integrity and improve performance.`
- Token IDs: `[48, 2722, 2497, 4218, 518, 321, 3539, 710, 563, 91, 323, 376, 555, 77, 71, 612, 87, 31, 1528, 87, 2497, 315, 81, 954, 518, 321, 3539, 348, 69, 273, 265, 3446, 339, 71, 379, 87, 87, 18, 225, 57, 430, 1528, 87, 510, 348, 77, 92, 650, 867, 3446, 1836, 778, 73, 75, 86, 1026, 3539, 315, 81, 2082, 1054, 1247, 696, 263, 379, 18]`
- Token strings: `['L', 'ists', 'Ġare', 'Ġmut', 'ab', 'le', 'Ġand', 'Ġal', 'lo', 'w', 'Ġm', 'od', 'if', 'i', 'c', 'ation', 's', ';', 'Ġtuple', 's', 'Ġare', 'Ġi', 'm', 'mut', 'ab', 'le', 'Ġand', 'Ġf', 'a', 'st', 'er', 'Ġto', 'Ġa', 'c', 'ce', 's', 's', '.', 'Ġ', 'U', 'se', 'Ġtuple', 's', 'Ġfor', 'Ġf', 'i', 'x', 'ed', 'Ġdata', 'Ġto', 'Ġensure', 'Ġint', 'e', 'g', 'r', 'ity', 'Ġand', 'Ġi', 'm', 'pro', 've', 'Ġper', 'form', 'an', 'ce', '.']`
- Decoded: `Lists are mutable and allow modifications; tuples are immutable and faster to access. Use tuples for fixed data to ensure integrity and improve performance.`
- Exact round-trip match: `True`
- Tokens per character: `0.4231`
- Suspicious-script tokens: `[]`

### Training text 23
- Original: `Bir JSONL dosyasında her satırın geçerli bir JSON nesnesi olup olmadığını nasıl doğrulayabilirim?`
- Token IDs: `[1104, 823, 48, 2696, 1246, 4465, 1358, 265, 295, 333, 823, 2475, 276, 1578, 1119, 347, 579, 271, 355, 4448, 35]`
- Token strings: `['Bir', 'ĠJSON', 'L', 'ĠdosyasÄ±nda', 'Ġher', 'ĠsatÄ±rÄ±n', 'ĠgeÃ§', 'er', 'li', 'Ġbir', 'ĠJSON', 'Ġnesne', 'si', 'Ġolup', 'ĠolmadÄ±ÄŁÄ±nÄ±', 'ĠnasÄ±l', 'ĠdoÄŁr', 'ul', 'ay', 'abilirim', '?']`
- Decoded: `Bir JSONL dosyasında her satırın geçerli bir JSON nesnesi olup olmadığını nasıl doğrulayabilirim?`
- Exact round-trip match: `True`
- Tokens per character: `0.2165`
- Suspicious-script tokens: `['ĠdosyasÄ±nda', 'ĠsatÄ±rÄ±n', 'ĠgeÃ§', 'ĠolmadÄ±ÄŁÄ±nÄ±', 'ĠnasÄ±l', 'ĠdoÄŁr']`

### Training text 24
- Original: `Her satırın geçerli bir JSON nesnesi olup olmadığını doğrulamak için satır satır JSON.parse() fonksiyonunu kullanabilirsin. Hata yakalama (try-catch) ile geçersiz formatlarda hata alırsan, o satırın geçersiz olduğunu anlarsın. Örnek: `try { JSON.parse(line); }`
- Token IDs: `[3313, 4465, 1358, 265, 295, 333, 823, 2475, 276, 1578, 1119, 2120, 474, 369, 607, 607, 823, 18, 2843, 430, 527, 626, 585, 702, 1180, 18, 4578, 2754, 328, 1705, 1330, 17, 71, 1570, 13, 432, 1358, 4093, 5030, 519, 3594, 2593, 16, 329, 4465, 1358, 4093, 1120, 449, 349, 684, 18, 3278, 30, 225, 68, 1330, 532, 823, 18, 2843, 430, 12, 773, 13, 31, 225, 97, 357, 1570, 1705, 73, 13, 532, 1517, 19, 1358, 4093, 225, 97, 68]`
- Token strings: `['Her', 'ĠsatÄ±rÄ±n', 'ĠgeÃ§', 'er', 'li', 'Ġbir', 'ĠJSON', 'Ġnesne', 'si', 'Ġolup', 'ĠolmadÄ±ÄŁÄ±nÄ±', 'ĠdoÄŁrula', 'mak', 'ĠiÃ§in', 'ĠsatÄ±r', 'ĠsatÄ±r', 'ĠJSON', '.', 'par', 'se', '()', 'Ġfonksiyon', 'unu', 'Ġkullan', 'abilirsin', '.', 'ĠHata', 'Ġyakala', 'ma', 'Ġ(', 'try', '-', 'c', 'atch', ')', 'Ġile', 'ĠgeÃ§', 'ersiz', 'Ġformatlarda', 'Ġhata', 'ĠalÄ±r', 'san', ',', 'Ġo', 'ĠsatÄ±rÄ±n', 'ĠgeÃ§', 'ersiz', 'ĠolduÄŁunu', 'Ġan', 'lar', 'sÄ±n', '.', 'ĠÃĸrnek', ':', 'Ġ', '`', 'try', 'Ġ{', 'ĠJSON', '.', 'par', 'se', '(', 'line', ')', ';', 'Ġ', '}', 'Ġc', 'atch', 'Ġ(', 'e', ')', 'Ġ{', 'Ġ/', '/', 'ĠgeÃ§', 'ersiz', 'Ġ', '}', '`']`
- Decoded: `Her satırın geçerli bir JSON nesnesi olup olmadığını doğrulamak için satır satır JSON.parse() fonksiyonunu kullanabilirsin. Hata yakalama (try-catch) ile geçersiz formatlarda hata alırsan, o satırın geçersiz olduğunu anlarsın. Örnek: `try { JSON.parse(line); }`
- Exact round-trip match: `True`
- Tokens per character: `0.2822`
- Suspicious-script tokens: `['ĠsatÄ±rÄ±n', 'ĠgeÃ§', 'ĠolmadÄ±ÄŁÄ±nÄ±', 'ĠdoÄŁrula', 'ĠiÃ§in', 'ĠsatÄ±r', 'ĠsatÄ±r', 'ĠgeÃ§', 'ĠalÄ±r', 'ĠsatÄ±rÄ±n', 'ĠgeÃ§', 'ĠolduÄŁunu', 'sÄ±n', 'ĠÃĸrnek', 'ĠgeÃ§']`

### Training text 25
- Original: `Job search süreci çok uzun ve moralim çok düştü. Ne yapmalıyım?`
- Token IDs: `[46, 83, 70, 3429, 275, 708, 4606, 1029, 1412, 338, 323, 290, 662, 81, 1029, 3860, 18, 1150, 1112, 35]`
- Token strings: `['J', 'o', 'b', 'Ġse', 'ar', 'ch', 'ĠsÃ¼reci', 'ĠÃ§ok', 'Ġuzun', 'Ġve', 'Ġm', 'or', 'ali', 'm', 'ĠÃ§ok', 'ĠdÃ¼ÅŁtÃ¼', '.', 'ĠNe', 'ĠyapmalÄ±yÄ±m', '?']`
- Decoded: `Job search süreci çok uzun ve moralim çok düştü. Ne yapmalıyım?`
- Exact round-trip match: `True`
- Tokens per character: `0.3175`
- Suspicious-script tokens: `['ĠsÃ¼reci', 'ĠÃ§ok', 'ĠÃ§ok', 'ĠdÃ¼ÅŁtÃ¼', 'ĠyapmalÄ±yÄ±m']`

### Training text 26
- Original: `Bu süreç gerçekten zor olabilir, ama her adım ilerleme demektir. Bugün bir CV güncellemesi yapmayı ve bir şirkete başvurmayı hedefle, bu sana yeniden enerji verebilir.`
- Token IDs: `[769, 3626, 2533, 2500, 868, 16, 728, 1246, 1540, 3689, 2944, 18, 530, 2826, 333, 409, 58, 1941, 657, 276, 2478, 338, 333, 925, 269, 79, 73, 351, 609, 90, 345, 1481, 1351, 321, 16, 959, 4753, 2699, 646, 265, 78, 77, 3002, 18]`
- Token strings: `['Bu', 'ĠsÃ¼reÃ§', 'ĠgerÃ§ekten', 'Ġzor', 'Ġolabilir', ',', 'Ġama', 'Ġher', 'ĠadÄ±m', 'Ġilerleme', 'Ġdemektir', '.', 'ĠBu', 'gÃ¼n', 'Ġbir', 'ĠC', 'V', 'ĠgÃ¼ncel', 'leme', 'si', 'ĠyapmayÄ±', 'Ġve', 'Ġbir', 'ĠÅŁ', 'ir', 'k', 'e', 'te', 'ĠbaÅŁ', 'v', 'ur', 'mayÄ±', 'Ġhedef', 'le', ',', 'Ġbu', 'Ġsana', 'Ġyeniden', 'Ġen', 'er', 'j', 'i', 'Ġverebilir', '.']`
- Decoded: `Bu süreç gerçekten zor olabilir, ama her adım ilerleme demektir. Bugün bir CV güncellemesi yapmayı ve bir şirkete başvurmayı hedefle, bu sana yeniden enerji verebilir.`
- Exact round-trip match: `True`
- Tokens per character: `0.2635`
- Suspicious-script tokens: `['ĠsÃ¼reÃ§', 'ĠgerÃ§ekten', 'ĠadÄ±m', 'gÃ¼n', 'ĠgÃ¼ncel', 'ĠyapmayÄ±', 'ĠÅŁ', 'ĠbaÅŁ', 'mayÄ±']`

### Training text 27
- Original: `A model's validation loss stops decreasing after 10 epochs, but training loss continues to drop. What does this suggest and what should I do next?`
- Token IDs: `[37, 372, 11, 87, 1380, 713, 3674, 87, 3470, 325, 301, 505, 339, 74, 1918, 1179, 324, 564, 708, 87, 16, 285, 453, 2760, 713, 357, 266, 88, 264, 545, 87, 3446, 307, 404, 84, 18, 2599, 1162, 557, 952, 3383, 597, 293, 89, 75, 3550, 3539, 2871, 311, 293, 76, 83, 271, 72, 753, 557, 373, 370, 35]`
- Token strings: `['A', 'Ġmodel', "'", 's', 'Ġvalidation', 'Ġloss', 'Ġstop', 's', 'Ġdec', 're', 'as', 'ing', 'Ġa', 'f', 'ter', 'Ġ10', 'Ġe', 'po', 'ch', 's', ',', 'Ġb', 'ut', 'Ġtraining', 'Ġloss', 'Ġc', 'on', 't', 'in', 'ue', 's', 'Ġto', 'Ġd', 'ro', 'p', '.', 'ĠW', 'hat', 'Ġdo', 'es', 'Ġth', 'is', 'Ġs', 'u', 'g', 'gest', 'Ġand', 'Ġwh', 'at', 'Ġs', 'h', 'o', 'ul', 'd', 'ĠI', 'Ġdo', 'Ġne', 'xt', '?']`
- Decoded: `A model's validation loss stops decreasing after 10 epochs, but training loss continues to drop. What does this suggest and what should I do next?`
- Exact round-trip match: `True`
- Tokens per character: `0.4041`
- Suspicious-script tokens: `[]`

### Training text 28
- Original: `This indicates overfitting. I would reduce model complexity, apply stronger regularization like dropout, or use early stopping based on validation loss to prevent the model from memorizing training data.`
- Token IDs: `[3324, 597, 412, 514, 71, 876, 87, 1719, 18, 753, 225, 91, 83, 271, 72, 670, 577, 379, 372, 945, 84, 321, 92, 1026, 16, 1485, 84, 1368, 1236, 2238, 265, 670, 75, 893, 385, 612, 437, 79, 73, 307, 404, 564, 453, 16, 2869, 754, 430, 324, 275, 1368, 3674, 84, 505, 285, 301, 650, 2600, 1380, 713, 3446, 472, 1068, 88, 3383, 73, 372, 348, 660, 386, 81, 290, 385, 505, 2760, 867, 18]`
- Token strings: `['Th', 'is', 'Ġin', 'di', 'c', 'ate', 's', 'Ġoverfitting', '.', 'ĠI', 'Ġ', 'w', 'o', 'ul', 'd', 'Ġre', 'du', 'ce', 'Ġmodel', 'Ġcom', 'p', 'le', 'x', 'ity', ',', 'Ġap', 'p', 'ly', 'Ġstr', 'ong', 'er', 'Ġre', 'g', 'ular', 'iz', 'ation', 'Ġli', 'k', 'e', 'Ġd', 'ro', 'po', 'ut', ',', 'Ġor', 'Ġu', 'se', 'Ġe', 'ar', 'ly', 'Ġstop', 'p', 'ing', 'Ġb', 'as', 'ed', 'Ġon', 'Ġvalidation', 'Ġloss', 'Ġto', 'Ġpr', 'even', 't', 'Ġth', 'e', 'Ġmodel', 'Ġf', 'rom', 'Ġme', 'm', 'or', 'iz', 'ing', 'Ġtraining', 'Ġdata', '.']`
- Decoded: `This indicates overfitting. I would reduce model complexity, apply stronger regularization like dropout, or use early stopping based on validation loss to prevent the model from memorizing training data.`
- Exact round-trip match: `True`
- Tokens per character: `0.3744`
- Suspicious-script tokens: `[]`

### Training text 29
- Original: `Compare the tradeoffs of storing data in JSON format versus normalized SQL tables for a product catalog with attributes like size, color, and weight. Which is better for frequent filtering by size?`
- Token IDs: `[39, 851, 2843, 73, 3383, 73, 313, 508, 270, 83, 4001, 87, 2931, 1333, 290, 505, 867, 412, 823, 2493, 482, 87, 487, 2251, 328, 295, 94, 650, 509, 53, 48, 4195, 2633, 510, 339, 472, 885, 357, 2912, 2840, 225, 91, 4016, 3071, 87, 437, 79, 73, 1996, 16, 3485, 290, 16, 3539, 225, 2861, 18, 2599, 76, 77, 708, 894, 285, 390, 1918, 510, 348, 325, 1106, 2074, 1141, 88, 418, 75, 285, 93, 1996, 35]`
- Token strings: `['C', 'om', 'par', 'e', 'Ġth', 'e', 'Ġt', 'ra', 'de', 'o', 'ff', 's', 'Ġof', 'Ġst', 'or', 'ing', 'Ġdata', 'Ġin', 'ĠJSON', 'Ġformat', 'Ġver', 's', 'us', 'Ġnor', 'ma', 'li', 'z', 'ed', 'ĠS', 'Q', 'L', 'Ġtab', 'les', 'Ġfor', 'Ġa', 'Ġpr', 'oduct', 'Ġc', 'atal', 'og', 'Ġ', 'w', 'ith', 'Ġattribute', 's', 'Ġli', 'k', 'e', 'Ġsize', ',', 'Ġcol', 'or', ',', 'Ġand', 'Ġ', 'weight', '.', 'ĠW', 'h', 'i', 'ch', 'Ġis', 'Ġb', 'et', 'ter', 'Ġfor', 'Ġf', 're', 'qu', 'ent', 'Ġfil', 't', 'erin', 'g', 'Ġb', 'y', 'Ġsize', '?']`
- Decoded: `Compare the tradeoffs of storing data in JSON format versus normalized SQL tables for a product catalog with attributes like size, color, and weight. Which is better for frequent filtering by size?`
- Exact round-trip match: `True`
- Tokens per character: `0.3959`
- Suspicious-script tokens: `[]`

### Training text 30
- Original: `JSON is better for schema flexibility and fast reads when filtering by size, as it avoids joins. However, SQL normalization supports better query performance and data integrity. Use JSON if size filtering is frequent and schema changes often.`
- Token IDs: `[812, 798, 894, 285, 390, 1918, 510, 293, 708, 73, 328, 348, 321, 92, 77, 354, 1026, 3539, 348, 69, 273, 4493, 87, 2871, 279, 1141, 88, 418, 75, 285, 93, 1996, 16, 1235, 315, 88, 339, 90, 83, 1698, 87, 225, 3837, 87, 18, 802, 2841, 479, 265, 16, 509, 53, 48, 3193, 293, 637, 618, 87, 285, 390, 1918, 3844, 1247, 696, 263, 379, 3539, 867, 778, 73, 75, 86, 1026, 18, 225, 57, 430, 823, 559, 1996, 1141, 88, 418, 75, 894, 348, 325, 1106, 2074, 3539, 293, 708, 73, 328, 357, 3350, 483, 87, 2931, 625, 18]`
- Token strings: `['JS', 'ON', 'Ġis', 'Ġb', 'et', 'ter', 'Ġfor', 'Ġs', 'ch', 'e', 'ma', 'Ġf', 'le', 'x', 'i', 'bil', 'ity', 'Ġand', 'Ġf', 'a', 'st', 'Ġread', 's', 'Ġwh', 'en', 'Ġfil', 't', 'erin', 'g', 'Ġb', 'y', 'Ġsize', ',', 'Ġas', 'Ġi', 't', 'Ġa', 'v', 'o', 'id', 's', 'Ġ', 'join', 's', '.', 'ĠH', 'ow', 'ev', 'er', ',', 'ĠS', 'Q', 'L', 'Ġnormalization', 'Ġs', 'up', 'port', 's', 'Ġb', 'et', 'ter', 'Ġquery', 'Ġper', 'form', 'an', 'ce', 'Ġand', 'Ġdata', 'Ġint', 'e', 'g', 'r', 'ity', '.', 'Ġ', 'U', 'se', 'ĠJSON', 'Ġif', 'Ġsize', 'Ġfil', 't', 'erin', 'g', 'Ġis', 'Ġf', 're', 'qu', 'ent', 'Ġand', 'Ġs', 'ch', 'e', 'ma', 'Ġc', 'han', 'ge', 's', 'Ġof', 'ten', '.']`
- Decoded: `JSON is better for schema flexibility and fast reads when filtering by size, as it avoids joins. However, SQL normalization supports better query performance and data integrity. Use JSON if size filtering is frequent and schema changes often.`
- Exact round-trip match: `True`
- Tokens per character: `0.4174`
- Suspicious-script tokens: `[]`

### Training text 31
- Original: `Entity Framework Core ile bir veritabanı işlemi yaparken, veri ekleme işlemi başarılı olmasına rağmen veritabanında görünmüyor. Bu durumda neleri kontrol etmeliyim?`
- Token IDs: `[41, 82, 88, 1026, 721, 86, 367, 91, 290, 79, 409, 290, 73, 432, 333, 482, 335, 518, 263, 261, 3042, 2099, 16, 410, 756, 657, 3042, 609, 275, 320, 261, 366, 2586, 4797, 482, 335, 518, 263, 578, 1856, 81, 268, 605, 18, 530, 1857, 296, 3480, 497, 1582, 1005, 35]`
- Token strings: `['E', 'n', 't', 'ity', 'ĠF', 'r', 'ame', 'w', 'or', 'k', 'ĠC', 'or', 'e', 'Ġile', 'Ġbir', 'Ġver', 'it', 'ab', 'an', 'Ä±', 'ĠiÅŁlemi', 'Ġyaparken', ',', 'Ġveri', 'Ġek', 'leme', 'ĠiÅŁlemi', 'ĠbaÅŁ', 'ar', 'Ä±l', 'Ä±', 'Ġol', 'masÄ±na', 'ĠraÄŁmen', 'Ġver', 'it', 'ab', 'an', 'Ä±nda', 'ĠgÃ¶rÃ¼n', 'm', 'Ã¼', 'yor', '.', 'ĠBu', 'Ġdurumda', 'Ġn', 'eleri', 'Ġkontrol', 'Ġetmeli', 'yim', '?']`
- Decoded: `Entity Framework Core ile bir veritabanı işlemi yaparken, veri ekleme işlemi başarılı olmasına rağmen veritabanında görünmüyor. Bu durumda neleri kontrol etmeliyim?`
- Exact round-trip match: `True`
- Tokens per character: `0.3171`
- Suspicious-script tokens: `['Ä±', 'ĠiÅŁlemi', 'ĠiÅŁlemi', 'ĠbaÅŁ', 'Ä±l', 'Ä±', 'masÄ±na', 'ĠraÄŁmen', 'Ä±nda', 'ĠgÃ¶rÃ¼n', 'Ã¼']`

### Training text 32
- Original: `İlk olarak, `SaveChanges()` metodunun çağrıldığını ve başarılı bir şekilde döndüğünü doğrula. Eğer işlem tamamlandıysa, `DbContext` örneğinin yaşam süresinin yeterli olup olmadığını kontrol et. Ayrıca, veritabanı içindeki tabloda veri eklenmeyebilir çünkü `Tra`
- Token IDs: `[1842, 1015, 16, 225, 68, 55, 919, 73, 39, 3350, 483, 87, 527, 68, 2284, 2640, 2272, 427, 1571, 338, 609, 275, 320, 261, 333, 1155, 473, 3993, 1146, 2120, 18, 3100, 695, 5149, 4067, 16, 225, 68, 40, 70, 39, 266, 415, 68, 448, 3340, 1237, 3460, 550, 857, 1818, 1599, 1578, 1119, 497, 584, 18, 225, 793, 86, 261, 1697, 16, 482, 335, 518, 263, 261, 1725, 4195, 563, 368, 410, 756, 471, 4118, 359, 4276, 225, 68, 56, 490, 388, 3025, 68, 592, 225, 68, 55, 919, 73, 39, 3350, 483, 87, 527, 68, 4420, 868, 18]`
- Token strings: `['Ä°lk', 'Ġolarak', ',', 'Ġ', '`', 'S', 'av', 'e', 'C', 'han', 'ge', 's', '()', '`', 'Ġmetod', 'unun', 'ĠÃ§a', 'ÄŁr', 'Ä±ldÄ±ÄŁÄ±nÄ±', 'Ġve', 'ĠbaÅŁ', 'ar', 'Ä±l', 'Ä±', 'Ġbir', 'ĠÅŁekilde', 'ĠdÃ¶n', 'dÃ¼ÄŁ', 'Ã¼nÃ¼', 'ĠdoÄŁrula', '.', 'ĠEÄŁer', 'ĠiÅŁlem', 'ĠtamamlandÄ±', 'ysa', ',', 'Ġ', '`', 'D', 'b', 'C', 'on', 'text', '`', 'ĠÃ¶rn', 'eÄŁ', 'inin', 'ĠyaÅŁ', 'am', 'ĠsÃ¼re', 'sinin', 'Ġyeterli', 'Ġolup', 'ĠolmadÄ±ÄŁÄ±nÄ±', 'Ġkontrol', 'Ġet', '.', 'Ġ', 'Ay', 'r', 'Ä±', 'ca', ',', 'Ġver', 'it', 'ab', 'an', 'Ä±', 'ĠiÃ§indeki', 'Ġtab', 'lo', 'da', 'Ġveri', 'Ġek', 'len', 'meye', 'bilir', 'ĠÃ§Ã¼nkÃ¼', 'Ġ', '`', 'T', 'ran', 'sa', 'ction', '`', 'Ġveya', 'Ġ', '`', 'S', 'av', 'e', 'C', 'han', 'ge', 's', '()', '`', 'ĠhatalÄ±', 'Ġolabilir', '.']`
- Decoded: `İlk olarak, `SaveChanges()` metodunun çağrıldığını ve başarılı bir şekilde döndüğünü doğrula. Eğer işlem tamamlandıysa, `DbContext` örneğinin yaşam süresinin yeterli olup olmadığını kontrol et. Ayrıca, veritabanı içindeki tabloda veri eklenmeyebilir çünkü `Tra`
- Exact round-trip match: `True`
- Tokens per character: `0.3257`
- Suspicious-script tokens: `['Ä°lk', 'ĠÃ§a', 'ÄŁr', 'Ä±ldÄ±ÄŁÄ±nÄ±', 'ĠbaÅŁ', 'Ä±l', 'Ä±', 'ĠÅŁekilde', 'ĠdÃ¶n', 'dÃ¼ÄŁ', 'Ã¼nÃ¼', 'ĠdoÄŁrula', 'ĠEÄŁer', 'ĠiÅŁlem', 'ĠtamamlandÄ±', 'ĠÃ¶rn', 'eÄŁ', 'ĠyaÅŁ', 'ĠsÃ¼re', 'ĠolmadÄ±ÄŁÄ±nÄ±', 'Ä±', 'Ä±', 'ĠiÃ§indeki', 'ĠÃ§Ã¼nkÃ¼', 'ĠhatalÄ±']`

### Training text 33
- Original: `Python'da bir listeyi sırayla dönen bir for döngüsü yaz. Listeyi yazdırmadan önce her elemanın karesini almak istiyorum. Hata alıyorum, nerede yanlış yapıyorum?`
- Token IDs: `[1900, 322, 69, 333, 1648, 2140, 473, 279, 333, 510, 916, 374, 18, 2030, 531, 1307, 1804, 1260, 1246, 692, 318, 3065, 1402, 2262, 2214, 18, 4578, 1186, 16, 4170, 270, 1172, 897, 1011, 35]`
- Token strings: `['Python', "'d", 'a', 'Ġbir', 'Ġlisteyi', 'ĠsÄ±rayla', 'ĠdÃ¶n', 'en', 'Ġbir', 'Ġfor', 'ĠdÃ¶ngÃ¼sÃ¼', 'Ġyaz', '.', 'ĠListe', 'yi', 'ĠyazdÄ±r', 'madan', 'ĠÃ¶nce', 'Ġher', 'Ġeleman', 'Ä±n', 'Ġkaresini', 'Ġalmak', 'Ġist', 'iyorum', '.', 'ĠHata', 'ĠalÄ±yorum', ',', 'Ġnere', 'de', 'ĠyanlÄ±ÅŁ', 'ĠyapÄ±', 'yorum', '?']`
- Decoded: `Python'da bir listeyi sırayla dönen bir for döngüsü yaz. Listeyi yazdırmadan önce her elemanın karesini almak istiyorum. Hata alıyorum, nerede yanlış yapıyorum?`
- Exact round-trip match: `True`
- Tokens per character: `0.2188`
- Suspicious-script tokens: `['ĠsÄ±rayla', 'ĠdÃ¶n', 'ĠdÃ¶ngÃ¼sÃ¼', 'ĠyazdÄ±r', 'ĠÃ¶nce', 'Ä±n', 'ĠalÄ±yorum', 'ĠyanlÄ±ÅŁ', 'ĠyapÄ±']`

### Training text 34
- Original: `For döngüsünde her elemanı karesine çevirip ekrana yazdırmak için `for` döngüsü kullanmalısın. Her elemanı `x**2` ile karesini alıp `print()` ile ekrana yazdırabilirsin. Örnek: `for x in liste: print(x**2)`.`
- Token IDs: `[42, 290, 4599, 1246, 2131, 804, 73, 1393, 2178, 469, 4952, 1307, 474, 369, 225, 68, 528, 68, 916, 1841, 18, 2143, 2131, 225, 68, 92, 14, 14, 22, 68, 432, 3065, 2350, 225, 68, 326, 527, 68, 432, 4952, 1307, 1180, 18, 3278, 30, 225, 68, 528, 225, 92, 412, 493, 30, 457, 12, 92, 14, 14, 22, 13, 68, 18]`
- Token strings: `['F', 'or', 'ĠdÃ¶ngÃ¼sÃ¼nde', 'Ġher', 'ĠelemanÄ±', 'Ġkar', 'e', 'sine', 'ĠÃ§evir', 'ip', 'Ġekrana', 'ĠyazdÄ±r', 'mak', 'ĠiÃ§in', 'Ġ', '`', 'for', '`', 'ĠdÃ¶ngÃ¼sÃ¼', 'ĠkullanmalÄ±sÄ±n', '.', 'ĠHer', 'ĠelemanÄ±', 'Ġ', '`', 'x', '*', '*', '2', '`', 'Ġile', 'Ġkaresini', 'ĠalÄ±p', 'Ġ', '`', 'print', '()', '`', 'Ġile', 'Ġekrana', 'ĠyazdÄ±r', 'abilirsin', '.', 'ĠÃĸrnek', ':', 'Ġ', '`', 'for', 'Ġ', 'x', 'Ġin', 'Ġliste', ':', 'Ġprint', '(', 'x', '*', '*', '2', ')', '`', '.']`
- Decoded: `For döngüsünde her elemanı karesine çevirip ekrana yazdırmak için `for` döngüsü kullanmalısın. Her elemanı `x**2` ile karesini alıp `print()` ile ekrana yazdırabilirsin. Örnek: `for x in liste: print(x**2)`.`
- Exact round-trip match: `True`
- Tokens per character: `0.2995`
- Suspicious-script tokens: `['ĠdÃ¶ngÃ¼sÃ¼nde', 'ĠelemanÄ±', 'ĠÃ§evir', 'ĠyazdÄ±r', 'ĠiÃ§in', 'ĠdÃ¶ngÃ¼sÃ¼', 'ĠkullanmalÄ±sÄ±n', 'ĠelemanÄ±', 'ĠalÄ±p', 'ĠyazdÄ±r', 'ĠÃĸrnek']`

### Training text 35
- Original: `Ben DarkMind miyim? Eğer öyleysem, ChatGPT, OpenAI veya Qwen değilim, değil mi?`
- Token IDs: `[38, 279, 543, 3622, 35, 3100, 356, 835, 2862, 392, 16, 1870, 16, 225, 51, 84, 279, 37, 45, 592, 225, 53, 91, 279, 740, 344, 16, 740, 821, 35]`
- Token strings: `['B', 'en', 'ĠDarkMind', 'Ġmiyim', '?', 'ĠEÄŁer', 'ĠÃ¶', 'yle', 'ys', 'em', ',', 'ĠChatGPT', ',', 'Ġ', 'O', 'p', 'en', 'A', 'I', 'Ġveya', 'Ġ', 'Q', 'w', 'en', 'ĠdeÄŁil', 'im', ',', 'ĠdeÄŁil', 'Ġmi', '?']`
- Decoded: `Ben DarkMind miyim? Eğer öyleysem, ChatGPT, OpenAI veya Qwen değilim, değil mi?`
- Exact round-trip match: `True`
- Tokens per character: `0.3797`
- Suspicious-script tokens: `['ĠEÄŁer', 'ĠÃ¶', 'ĠdeÄŁil', 'ĠdeÄŁil']`

### Training text 36
- Original: `Evet, ben DarkMind, Tarık Yasin Sağlıcak tarafından geliştirilen Türkçe odaklı bir yazılım asistanıyım. Ben ChatGPT, OpenAI veya Qwen değilim.`
- Token IDs: `[41, 90, 390, 16, 285, 279, 543, 16, 382, 275, 465, 795, 301, 264, 509, 69, 287, 421, 2423, 2390, 1174, 3720, 548, 1510, 333, 1640, 339, 276, 300, 261, 968, 18, 977, 1870, 16, 225, 51, 84, 279, 37, 45, 592, 225, 53, 91, 279, 740, 344, 18]`
- Token strings: `['E', 'v', 'et', ',', 'Ġb', 'en', 'ĠDarkMind', ',', 'ĠT', 'ar', 'Ä±k', 'ĠY', 'as', 'in', 'ĠS', 'a', 'ÄŁ', 'lÄ±', 'cak', 'Ġtaraf', 'Ä±ndan', 'ĠgeliÅŁtirilen', 'ĠTÃ¼rkÃ§e', 'ĠodaklÄ±', 'Ġbir', 'ĠyazÄ±lÄ±m', 'Ġa', 'si', 'stan', 'Ä±', 'yÄ±m', '.', 'ĠBen', 'ĠChatGPT', ',', 'Ġ', 'O', 'p', 'en', 'A', 'I', 'Ġveya', 'Ġ', 'Q', 'w', 'en', 'ĠdeÄŁil', 'im', '.']`
- Decoded: `Evet, ben DarkMind, Tarık Yasin Sağlıcak tarafından geliştirilen Türkçe odaklı bir yazılım asistanıyım. Ben ChatGPT, OpenAI veya Qwen değilim.`
- Exact round-trip match: `True`
- Tokens per character: `0.3451`
- Suspicious-script tokens: `['Ä±k', 'ÄŁ', 'lÄ±', 'Ä±ndan', 'ĠgeliÅŁtirilen', 'ĠTÃ¼rkÃ§e', 'ĠodaklÄ±', 'ĠyazÄ±lÄ±m', 'Ä±', 'yÄ±m', 'ĠdeÄŁil']`

### Training text 37
- Original: `Bir ASP.NET Core Web API projesinde, kullanıcı kaydı yaparken bir POST isteği geldiğinde, kontrolörde ilk olarak neyi doğrulamalıyım?`
- Token IDs: `[1104, 995, 55, 52, 18, 50, 41, 56, 409, 290, 73, 2599, 3995, 995, 52, 45, 1651, 16, 996, 907, 503, 2099, 333, 309, 51, 55, 56, 2262, 814, 2257, 1348, 1788, 16, 497, 837, 270, 2510, 1015, 373, 531, 2120, 982, 35]`
- Token strings: `['Bir', 'ĠA', 'S', 'P', '.', 'N', 'E', 'T', 'ĠC', 'or', 'e', 'ĠW', 'eb', 'ĠA', 'P', 'I', 'Ġprojesinde', ',', 'ĠkullanÄ±cÄ±', 'Ġkay', 'dÄ±', 'Ġyaparken', 'Ġbir', 'ĠP', 'O', 'S', 'T', 'Ġist', 'eÄŁi', 'Ġgel', 'diÄŁ', 'inde', ',', 'Ġkontrol', 'Ã¶r', 'de', 'Ġilk', 'Ġolarak', 'Ġne', 'yi', 'ĠdoÄŁrula', 'malÄ±yÄ±m', '?']`
- Decoded: `Bir ASP.NET Core Web API projesinde, kullanıcı kaydı yaparken bir POST isteği geldiğinde, kontrolörde ilk olarak neyi doğrulamalıyım?`
- Exact round-trip match: `True`
- Tokens per character: `0.3233`
- Suspicious-script tokens: `['ĠkullanÄ±cÄ±', 'dÄ±', 'eÄŁi', 'diÄŁ', 'Ã¶r', 'ĠdoÄŁrula', 'malÄ±yÄ±m']`

### Training text 38
- Original: `Kullanıcı adı, e-posta ve şifre gibi gerekli alanların eksik olmamasını kontrol etmeliyim. Ayrıca e-posta formatının geçerli olup olmadığını doğrulamalıyım. Bu doğrulamalar, ModelState.IsValid ile yapılabilir.`
- Token IDs: `[302, 1805, 16, 324, 17, 564, 273, 69, 338, 4644, 1024, 3619, 2941, 858, 756, 276, 79, 627, 1280, 497, 1582, 1005, 18, 225, 793, 86, 261, 1697, 324, 17, 564, 273, 69, 2493, 1529, 1358, 265, 295, 1578, 1119, 2120, 982, 18, 530, 579, 271, 550, 805, 16, 1080, 55, 88, 876, 18, 45, 87, 58, 782, 432, 1217, 18]`
- Token strings: `['KullanÄ±cÄ±', 'ĠadÄ±', ',', 'Ġe', '-', 'po', 'st', 'a', 'Ġve', 'ĠÅŁifre', 'Ġgibi', 'Ġgerekli', 'Ġalan', 'larÄ±n', 'Ġek', 'si', 'k', 'Ġolma', 'masÄ±nÄ±', 'Ġkontrol', 'Ġetmeli', 'yim', '.', 'Ġ', 'Ay', 'r', 'Ä±', 'ca', 'Ġe', '-', 'po', 'st', 'a', 'Ġformat', 'Ä±nÄ±n', 'ĠgeÃ§', 'er', 'li', 'Ġolup', 'ĠolmadÄ±ÄŁÄ±nÄ±', 'ĠdoÄŁrula', 'malÄ±yÄ±m', '.', 'ĠBu', 'ĠdoÄŁr', 'ul', 'am', 'alar', ',', 'ĠModel', 'S', 't', 'ate', '.', 'I', 's', 'V', 'alid', 'Ġile', 'ĠyapÄ±labilir', '.']`
- Decoded: `Kullanıcı adı, e-posta ve şifre gibi gerekli alanların eksik olmamasını kontrol etmeliyim. Ayrıca e-posta formatının geçerli olup olmadığını doğrulamalıyım. Bu doğrulamalar, ModelState.IsValid ile yapılabilir.`
- Exact round-trip match: `True`
- Tokens per character: `0.2919`
- Suspicious-script tokens: `['KullanÄ±cÄ±', 'ĠadÄ±', 'ĠÅŁifre', 'larÄ±n', 'masÄ±nÄ±', 'Ä±', 'Ä±nÄ±n', 'ĠgeÃ§', 'ĠolmadÄ±ÄŁÄ±nÄ±', 'ĠdoÄŁrula', 'malÄ±yÄ±m', 'ĠdoÄŁr', 'ĠyapÄ±labilir']`

### Training text 39
- Original: `Python'da bir listedeki çift sayıları filtrelemek için en basit yöntem nedir? Hem list comprehension hem de filter fonksiyonunu kullanarak örnek ver.`
- Token IDs: `[1900, 322, 69, 333, 1253, 1577, 1356, 2356, 369, 646, 767, 2753, 599, 35, 802, 392, 440, 1036, 1575, 342, 4693, 626, 585, 2505, 516, 482, 18]`
- Token strings: `['Python', "'d", 'a', 'Ġbir', 'Ġlistedeki', 'ĠÃ§ift', 'ĠsayÄ±larÄ±', 'Ġfiltrelemek', 'ĠiÃ§in', 'Ġen', 'Ġbasit', 'ĠyÃ¶ntem', 'Ġnedir', '?', 'ĠH', 'em', 'Ġlist', 'Ġcomprehension', 'Ġhem', 'Ġde', 'Ġfilter', 'Ġfonksiyon', 'unu', 'Ġkullanarak', 'ĠÃ¶rnek', 'Ġver', '.']`
- Decoded: `Python'da bir listedeki çift sayıları filtrelemek için en basit yöntem nedir? Hem list comprehension hem de filter fonksiyonunu kullanarak örnek ver.`
- Exact round-trip match: `True`
- Tokens per character: `0.1812`
- Suspicious-script tokens: `['ĠÃ§ift', 'ĠsayÄ±larÄ±', 'ĠiÃ§in', 'ĠyÃ¶ntem', 'ĠÃ¶rnek']`

### Training text 40
- Original: `Çift sayıları filtrelemek için `list comprehension` veya `filter` fonksiyonu kullanılabilir. List comprehension ile: `cift_sayilar = [x for x in liste if x % 2 == 0]`. Filter ile: `cift_sayilar = list(filter(lambda x: x % 2 == 0, liste))`. Her ikisi de basit v`
- Token IDs: `[5114, 1356, 2356, 369, 225, 68, 3432, 1036, 68, 592, 225, 68, 5284, 68, 1656, 664, 18, 1448, 1036, 432, 30, 225, 68, 544, 74, 88, 67, 87, 355, 288, 275, 308, 417, 92, 510, 225, 92, 412, 493, 559, 225, 92, 994, 444, 602, 541, 65, 68, 18, 721, 288, 1918, 432, 30, 225, 68, 544, 74, 88, 67, 87, 355, 288, 275, 308, 440, 12, 5284, 12, 1050, 225, 92, 30, 225, 92, 994, 444, 602, 541, 16, 493, 697, 68, 18, 2143, 1173, 276, 342, 767, 338, 768, 398, 2102, 18]`
- Token strings: `['Ãĩift', 'ĠsayÄ±larÄ±', 'Ġfiltrelemek', 'ĠiÃ§in', 'Ġ', '`', 'list', 'Ġcomprehension', '`', 'Ġveya', 'Ġ', '`', 'filter', '`', 'Ġfonksiyonu', 'ĠkullanÄ±labilir', '.', 'ĠList', 'Ġcomprehension', 'Ġile', ':', 'Ġ', '`', 'ci', 'f', 't', '_', 's', 'ay', 'il', 'ar', 'Ġ=', 'Ġ[', 'x', 'Ġfor', 'Ġ', 'x', 'Ġin', 'Ġliste', 'Ġif', 'Ġ', 'x', 'Ġ%', 'Ġ2', 'Ġ==', 'Ġ0', ']', '`', '.', 'ĠF', 'il', 'ter', 'Ġile', ':', 'Ġ', '`', 'ci', 'f', 't', '_', 's', 'ay', 'il', 'ar', 'Ġ=', 'Ġlist', '(', 'filter', '(', 'lambda', 'Ġ', 'x', ':', 'Ġ', 'x', 'Ġ%', 'Ġ2', 'Ġ==', 'Ġ0', ',', 'Ġliste', '))', '`', '.', 'ĠHer', 'Ġiki', 'si', 'Ġde', 'Ġbasit', 'Ġve', 'Ġokun', 'ak', 'lÄ±dÄ±r', '.']`
- Decoded: `Çift sayıları filtrelemek için `list comprehension` veya `filter` fonksiyonu kullanılabilir. List comprehension ile: `cift_sayilar = [x for x in liste if x % 2 == 0]`. Filter ile: `cift_sayilar = list(filter(lambda x: x % 2 == 0, liste))`. Her ikisi de basit v`
- Exact round-trip match: `True`
- Tokens per character: `0.3431`
- Suspicious-script tokens: `['Ãĩift', 'ĠsayÄ±larÄ±', 'ĠiÃ§in', 'ĠkullanÄ±labilir', 'lÄ±dÄ±r']`


## Repository Evidence

### Tokenizer history
```text
1f36e97 Build DarkMind data and evaluation pipeline
2654f21 Approve second self-improvement candidates
e75e596 Approve self-improvement candidates and retrain tokenizer
4bdf0e8 Add self-improvement evaluation pipeline
06f9069 Add basic chat and fallback dataset
1cdec50 Bootstrap DarkMind data pipeline
9567b11 Improve targeted dialogue examples
662fb81 Expand DarkMind corpus v0.3
da3350e Update DarkMind corpus v0.2 and tokenizer
0f73ccc Initial DarkMind tiny GPT pipeline
```

### Tokenizer mentions in history/config/docs
```text
git command failed: 'charmap' codec can't decode byte 0x81 in position 13177: character maps to <undefined>
```

### Recent commits touching tokenizer files
```text
1f36e97 Build DarkMind data and evaluation pipeline
 tokenizer/darkmind-tokenizer/merges.txt | 7248 ++++++++++++++++---------------
 tokenizer/darkmind-tokenizer/vocab.json |    2 +-
 2 files changed, 3819 insertions(+), 3431 deletions(-)
2654f21 Approve second self-improvement candidates
 tokenizer/darkmind-tokenizer/merges.txt | 2840 +++++++++++++++----------------
 tokenizer/darkmind-tokenizer/vocab.json |    2 +-
 2 files changed, 1421 insertions(+), 1421 deletions(-)
e75e596 Approve self-improvement candidates and retrain tokenizer
 tokenizer/darkmind-tokenizer/merges.txt | 4146 ++++++++++++++++---------------
 tokenizer/darkmind-tokenizer/vocab.json |    2 +-
 2 files changed, 2125 insertions(+), 2023 deletions(-)
4bdf0e8 Add self-improvement evaluation pipeline
 tokenizer/darkmind-tokenizer/merges.txt | 6537 +++++++++++++++++++------------
 tokenizer/darkmind-tokenizer/vocab.json |    2 +-
 2 files changed, 3979 insertions(+), 2560 deletions(-)
06f9069 Add basic chat and fallback dataset
 tokenizer/darkmind-tokenizer/merges.txt | 3169 ++++++++++++++++---------------
 tokenizer/darkmind-tokenizer/vocab.json |    2 +-
 2 files changed, 1685 insertions(+), 1486 deletions(-)
9567b11 Improve targeted dialogue examples
 tokenizer/darkmind-tokenizer/merges.txt | 3502 ++++++++++++++++---------------
 tokenizer/darkmind-tokenizer/vocab.json |    2 +-
 2 files changed, 1855 insertions(+), 1649 deletions(-)
662fb81 Expand DarkMind corpus v0.3
 tokenizer/darkmind-tokenizer/merges.txt | 3839 +++++++++++++++++++++----------
 tokenizer/darkmind-tokenizer/vocab.json |    2 +-
 2 files changed, 2680 insertions(+), 1161 deletions(-)
da3350e Update DarkMind corpus v0.2 and tokenizer
 tokenizer/darkmind-tokenizer/merges.txt | 1638 ++++++++++++++++++++++++-------
 tokenizer/darkmind-tokenizer/vocab.json |    2 +-
 2 files changed, 1262 insertions(+), 378 deletions(-)
0f73ccc Initial DarkMind tiny GPT pipeline
 tokenizer/darkmind-tokenizer/merges.txt | 445 ++++++++++++++++++++++++++++++++
 tokenizer/darkmind-tokenizer/vocab.json |   1 +
 2 files changed, 446 insertions(+)
```
