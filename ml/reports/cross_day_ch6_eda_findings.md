# Channel-6 (2026-09-15/16/17) cross-day EDA findings

Generated from 81 channel-6 sessions across ['2026-09-15', '2026-09-16', '2026-09-17'] (`ml/evaluation/cross_day_eda_ch6.py`). Every number below is measured directly from this repo's own `data/` for the exact subset the BiLSTM cross-day track uses -- nothing here is assumed from metadata.json or prior docs.

## 1. Session counts by date/label/person

```
date        label         person_id
2026-09-15  authorized    anjali        7
                          barath        7
            none                       10
            unauthorized  abdul         2
                          divya         2
                          harshitha     2
                          siva          2
                          sumanth       2
2026-09-16  authorized    anjali        5
                          barath       10
            none                        4
            unauthorized  divya         2
                          kishore       2
                          manas         2
                          sumanth       2
2026-09-17  authorized    anjali        6
                          barath        4
            none                        3
            unauthorized  divya         3
                          harshtiha     2
                          manas         2
```

## 2. Channel / bandwidth / PHY configuration

Unique values observed per date (all channel-6-filtered sessions already share channel_primary=6 by construction -- this checks everything else):

```
           channel_primary channel_freq_mhz  cwb  mcs stbc n_subcarriers
date                                                                    
2026-09-15             [6]           [2437]  [0]  [7]  [1]         [128]
2026-09-16             [6]           [2437]  [0]  [7]  [1]         [128]
2026-09-17             [6]           [2437]  [0]  [7]  [1]         [128]
```


Fraction of each session's packets kept after MAC+PHY-combo filtering (low values would mean the "one consistent CSI packet configuration" assumption is being violated mid-session):

```
            mean   min   max
date                        
2026-09-15 0.953 0.809 0.988
2026-09-16 0.983 0.876 0.998
2026-09-17 0.969 0.839 0.986
```


Sessions with more than one distinct `csi_len` (raw packet format changing mid-session, before MAC/PHY filtering): 79 / 81.

```
                                      session_dir              csi_len_dist
     authorized/2026-09-15/20260915_151030_anjali   {128: 1077, 256: 52218}
     authorized/2026-09-15/20260915_153434_anjali   {128: 1062, 256: 50390}
     authorized/2026-09-15/20260915_153933_anjali   {128: 1061, 256: 45577}
     authorized/2026-09-15/20260915_154548_barath   {128: 1002, 256: 44816}
     authorized/2026-09-15/20260915_155004_barath    {128: 809, 256: 36434}
     authorized/2026-09-15/20260915_160607_anjali    {128: 943, 256: 34185}
     authorized/2026-09-15/20260915_161212_anjali   {128: 1023, 256: 48318}
     authorized/2026-09-15/20260915_161737_barath    {128: 875, 256: 45892}
     authorized/2026-09-15/20260915_162230_barath    {128: 903, 256: 48367}
     authorized/2026-09-15/20260915_163025_anjali   {128: 1031, 256: 37850}
     authorized/2026-09-15/20260915_163456_anjali   {128: 1000, 256: 35771}
     authorized/2026-09-15/20260915_164124_barath    {128: 866, 256: 44295}
     authorized/2026-09-15/20260915_180952_barath    {128: 660, 256: 60347}
     authorized/2026-09-15/20260915_181508_barath    {128: 774, 256: 58384}
     authorized/2026-09-16/20260916_132313_anjali    {128: 788, 256: 60456}
     authorized/2026-09-16/20260916_132829_barath    {128: 796, 256: 60301}
     authorized/2026-09-16/20260916_133308_barath    {128: 862, 256: 61470}
     authorized/2026-09-16/20260916_142654_anjali   {128: 1127, 256: 57995}
     authorized/2026-09-16/20260916_144318_anjali      {128: 7, 256: 60523}
     authorized/2026-09-16/20260916_144821_barath      {128: 7, 256: 55678}
     authorized/2026-09-16/20260916_145304_barath     {128: 18, 256: 62658}
     authorized/2026-09-16/20260916_145921_anjali     {128: 19, 256: 59598}
     authorized/2026-09-16/20260916_150336_anjali      {128: 4, 256: 58515}
     authorized/2026-09-16/20260916_152145_barath      {128: 9, 256: 57525}
     authorized/2026-09-16/20260916_152556_barath      {128: 3, 256: 57149}
     authorized/2026-09-16/20260916_153337_barath     {128: 17, 256: 58709}
     authorized/2026-09-16/20260916_170839_barath    {128: 252, 256: 27746}
     authorized/2026-09-16/20260916_171125_barath    {128: 229, 256: 26311}
     authorized/2026-09-17/20260917_181415_anjali    {128: 609, 256: 27855}
     authorized/2026-09-17/20260917_181647_barath    {128: 500, 256: 29406}
     authorized/2026-09-17/20260917_181923_barath    {128: 533, 256: 30113}
     authorized/2026-09-17/20260917_182152_anjali    {128: 456, 256: 29024}
     authorized/2026-09-17/20260917_182354_anjali    {128: 492, 256: 29010}
     authorized/2026-09-17/20260917_183129_anjali    {128: 549, 256: 26042}
     authorized/2026-09-17/20260917_184714_anjali    {128: 381, 256: 25785}
     authorized/2026-09-17/20260917_184937_anjali    {128: 385, 256: 28895}
     authorized/2026-09-17/20260917_185221_barath    {128: 465, 256: 28757}
     authorized/2026-09-17/20260917_185506_barath    {128: 472, 256: 27896}
                  none/2026-09-15/20260915_150333   {128: 1235, 256: 26183}
                  none/2026-09-15/20260915_150550    {128: 466, 256: 23949}
                  none/2026-09-15/20260915_152301   {128: 1499, 256: 67882}
                  none/2026-09-15/20260915_152825   {128: 1129, 256: 44877}
                  none/2026-09-15/20260915_155425    {128: 860, 256: 38681}
                  none/2026-09-15/20260915_155941   {128: 1067, 256: 43243}
                  none/2026-09-15/20260915_175702    {128: 384, 256: 30211}
                  none/2026-09-15/20260915_180705    {128: 363, 256: 28632}
                  none/2026-09-15/20260915_182121    {128: 390, 256: 29652}
                  none/2026-09-15/20260915_182340    {128: 450, 256: 30270}
                  none/2026-09-16/20260916_131138    {128: 889, 256: 57993}
                  none/2026-09-16/20260916_131548    {128: 445, 256: 24468}
                  none/2026-09-16/20260916_131825    {128: 959, 256: 60020}
                  none/2026-09-16/20260916_133907 {128: 12643, 256: 723377}
                  none/2026-09-17/20260917_134430       {128: 14, 256: 626}
                  none/2026-09-17/20260917_134508 {128: 12423, 256: 627109}
                  none/2026-09-17/20260917_181241    {128: 245, 256: 14376}
    unauthorized/2026-09-15/20260915_165524_divya    {128: 570, 256: 29464}
    unauthorized/2026-09-15/20260915_165742_divya    {128: 527, 256: 29220}
unauthorized/2026-09-15/20260915_170022_harshitha    {128: 514, 256: 26554}
unauthorized/2026-09-15/20260915_170244_harshitha    {128: 470, 256: 30452}
  unauthorized/2026-09-15/20260915_170508_sumanth    {128: 425, 256: 27094}
  unauthorized/2026-09-15/20260915_170719_sumanth    {128: 597, 256: 29928}
    unauthorized/2026-09-15/20260915_171532_abdul    {128: 339, 256: 31800}
    unauthorized/2026-09-15/20260915_171755_abdul    {128: 381, 256: 31620}
     unauthorized/2026-09-15/20260915_175047_siva    {128: 335, 256: 31664}
     unauthorized/2026-09-15/20260915_175309_siva    {128: 388, 256: 30466}
    unauthorized/2026-09-16/20260916_151236_divya      {128: 6, 256: 58843}
    unauthorized/2026-09-16/20260916_151702_divya      {128: 2, 256: 57925}
  unauthorized/2026-09-16/20260916_154429_sumanth      {128: 3, 256: 28717}
    unauthorized/2026-09-16/20260916_154926_manas      {128: 8, 256: 31128}
    unauthorized/2026-09-16/20260916_155144_manas      {128: 2, 256: 27195}
  unauthorized/2026-09-16/20260916_155503_kishore      {128: 9, 256: 28247}
  unauthorized/2026-09-16/20260916_155718_kishore      {128: 2, 256: 27235}
    unauthorized/2026-09-17/20260917_183506_manas    {128: 505, 256: 26491}
    unauthorized/2026-09-17/20260917_183730_manas    {128: 529, 256: 24932}
    unauthorized/2026-09-17/20260917_183958_divya    {128: 414, 256: 29602}
    unauthorized/2026-09-17/20260917_184204_divya    {128: 426, 256: 29167}
    unauthorized/2026-09-17/20260917_184432_divya    {128: 549, 256: 29523}
unauthorized/2026-09-17/20260917_190415_harshtiha    {128: 377, 256: 28927}
unauthorized/2026-09-17/20260917_190642_harshtiha    {128: 411, 256: 30803}
```

## 3. MAC address routing

`frac_to_board_mac`: fraction of a session's RAW packets (before filtering) addressed to that session's own ESP32 (`dst_mac == board_mac`).

```
           frac_to_board_mac             n_distinct_mac_pairs        
                        mean   min   max                 mean min max
date                                                                 
2026-09-15             0.954 0.810 0.989                3.088   3   5
2026-09-16             0.984 0.879 0.999                3.074   3   4
2026-09-17             0.970 0.841 0.987                2.500   2   5
```


Distinct `board_mac` values across all 81 sessions: 1 (['ac:27:6e:a5:5b:c8']) -- same physical ESP32 used throughout.

## 4. Packet-rate / timing differences

Native packet rate (Hz), post MAC/PHY filtering, per date:

```
            mean  std   min   max
date                             
2026-09-15 210.2 41.9 130.5 268.6
2026-09-16 243.8  8.3 225.7 255.5
2026-09-17 236.5 21.3 165.4 261.8
```


Same, broken down by label -- this is the exact shape of the previously-found packet-rate/window-duration confound (rate correlating with label within a date). The BiLSTM pipeline's `time_resample.py` step already resamples every session onto one common rate before windowing specifically to neutralize this, but the raw numbers below are worth checking directly:

```
                         mean   min   max  count
date       label                                
2026-09-15 authorized   188.3 135.2 253.4     14
           none         207.4 130.5 254.5     10
           unauthorized 243.8 228.0 268.6     10
2026-09-16 authorized   244.2 225.7 255.4     15
           none         249.3 244.1 255.4      4
           unauthorized 240.4 227.1 255.5      8
2026-09-17 authorized   239.0 217.2 250.0     10
           none         217.2 165.4 246.8      3
           unauthorized 241.3 208.4 261.8      7
```

## 5. Person-movement (motion) mix

```
date        label         motion  
2026-09-15  authorized    standing     5
                          walking      9
            none                      10
            unauthorized  standing     5
                          walking      5
2026-09-16  authorized    standing     6
                          walking      9
            none                       4
            unauthorized  standing     4
                          walking      4
2026-09-17  authorized    standing     5
                          walking      5
            none                       3
            unauthorized  standing     4
                          walking      3
```

## 6. Environment / AP placement notes (from metadata.json)

`ap_source` values per date:
```
date
2026-09-15    [home-router]
2026-09-16    [home-router]
2026-09-17    [home-router]
```


No session across these three dates has a non-empty `notes` field -- physical AP/ESP32 placement and any environmental changes between days are **not recorded anywhere in this dataset** and cannot be checked from data alone. This is a real, unfixable-after-the-fact gap: if placement moved between days, there is no metadata trail to detect or control for it, only the amplitude-drift numbers below as an indirect symptom.


Mean RSSI per date (a cheap proxy for gross link/placement/environment change -- a stable link across days should show similar RSSI; a shift suggests something physical changed, whether AP position, obstruction, or room occupancy):

```
            mean  std   min   max
date                             
2026-09-15 -40.7  2.6 -47.2 -37.6
2026-09-16 -42.7  2.4 -46.7 -39.0
2026-09-17 -41.6  2.2 -45.3 -37.6
```

## 7. Amplitude-distribution drift across days (raw, pre-denoising)

Per-subcarrier amplitude mean/std, pooled over every dominant packet of every channel-6 session for that date+label (NOT windowed -- uses all data for a tight estimate), compared pairwise across the three dates.


**none** (n sessions per date: {'2026-09-15': 10, '2026-09-16': 4, '2026-09-17': 3}):

```
                    pair  amp_mean |diff| (mean over subcarriers)  amp_mean |diff| (max over subcarriers)  amp_std |diff| (mean over subcarriers)  rel_mean_shift_%
2026-09-15 vs 2026-09-16                                   0.4575                                  5.1183                                  1.2309            4.7618
2026-09-15 vs 2026-09-17                                   1.4759                                  4.7277                                  0.5656           14.6825
2026-09-16 vs 2026-09-17                                   1.4583                                  8.5824                                  1.3755           14.4287
```


**authorized** (n sessions per date: {'2026-09-15': 14, '2026-09-16': 15, '2026-09-17': 10}):

```
                    pair  amp_mean |diff| (mean over subcarriers)  amp_mean |diff| (max over subcarriers)  amp_std |diff| (mean over subcarriers)  rel_mean_shift_%
2026-09-15 vs 2026-09-16                                   0.6704                                  1.9444                                  0.6583            6.1558
2026-09-15 vs 2026-09-17                                   1.5452                                  8.0232                                  0.5048           13.8503
2026-09-16 vs 2026-09-17                                   1.0380                                  6.3591                                  0.5868            9.1313
```


**unauthorized** (n sessions per date: {'2026-09-15': 10, '2026-09-16': 8, '2026-09-17': 7}):

```
                    pair  amp_mean |diff| (mean over subcarriers)  amp_mean |diff| (max over subcarriers)  amp_std |diff| (mean over subcarriers)  rel_mean_shift_%
2026-09-15 vs 2026-09-16                                   0.7122                                  3.2037                                  0.4337            6.7117
2026-09-15 vs 2026-09-17                                   2.3014                                 17.6105                                  0.6786           20.5312
2026-09-16 vs 2026-09-17                                   1.9720                                 14.4441                                  0.8064           17.3202
```
