# NUAA step zero: image statistics against the delivered scores (aggregate, no identifiers)

Run `20261103-STEPZERO-78ca9e4` (commit `78ca9e4`, dirty tree: no), 2026-11-03T15:56:16+0100. 12614 official NUAA images (16 subjects), 6000 cache frames; no training, no inference, no face detection. Every measure is descriptive: associations, not causes.

Statistics: `lum_mean` = mean of the OpenCV grey image (0.299 R + 0.587 G + 0.114 B, uint8), 0-255; `lum_std` = population standard deviation of the grey image, 0-255; `sat_mean` = mean of the S channel of the OpenCV HSV conversion, 0-255; `lap_var` = population variance of the 3x3 Laplacian of the grey image (cv2.Laplacian, CV_64F, ksize 1); `width` = image width in pixels (raw view only); `height` = image height in pixels (raw view only); `bytes` = size of the original file in bytes, from the manifest (raw view only).

Class AUROC: AUROC of the statistic as a class separator: label 1 = attack, higher statistic -> attack; Mann-Whitney U with ties counted 0.5; a value below 0.5 means bona fide frames have the higher statistic. Bootstrap: percentiles 2.5 and 97.5 of the class AUROC over 1,000 resamples with replacement of the subjects of the session (seed 0, the same resamples for every statistic); a resample without one of the classes does not count (valid_resamples). Within-subject AUROC: class AUROC computed inside each subject x session with both classes; median, minimum and maximum over those subjects, with their number (per-subject values in the restricted per_subject_auroc.csv). Spearman: Spearman rank correlation (average ranks for ties) between the statistic and score_attack; NA with fewer than 3 pairs or a constant variable.

Crop view: the crop view describes the network input after the reference enhancement chain (bilateral filter, CLAHE, unsharp mask, gamma and a final linear contrast/brightness step), not the face region of the original. All row: in the all row of a set with two sessions the sessions of a subject are pooled, for the within-subject AUROC and for the bootstrap alike.

## Sets, raw view (original images)

### official_train (3491 images)

| session | class | n | lum_mean median (Q1–Q3) | lum_std median (Q1–Q3) | sat_mean median (Q1–Q3) | lap_var median (Q1–Q3) | width median (Q1–Q3) | height median (Q1–Q3) | bytes median (Q1–Q3) |
|---|---|---|---|---|---|---|---|---|---|
| 01 | bona fide | 889 | 130.91 (127.06–135.64) | 82.42 (80.04–83.72) | 42.48 (38.41–46.51) | 339.56 (311.75–411.64) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 30376.00 (29700.00–34149.00) |
| 01 | attack | 855 | 112.91 (105.64–119.55) | 57.45 (53.45–63.06) | 82.46 (70.57–89.40) | 105.92 (64.28–136.09) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 26274.00 (24024.50–28086.00) |
| 02 | bona fide | 854 | 117.62 (110.38–126.27) | 75.46 (71.18–77.51) | 65.19 (60.28–74.15) | 176.51 (164.02–218.37) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 31125.50 (29986.50–32507.75) |
| 02 | attack | 893 | 122.51 (113.71–133.18) | 61.08 (57.76–63.88) | 47.89 (41.68–54.22) | 155.73 (110.07–230.22) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 31080.00 (28347.00–33571.00) |
| all | bona fide | 1743 | 126.84 (117.12–132.60) | 79.45 (75.55–82.79) | 50.16 (42.35–64.95) | 252.08 (169.69–340.59) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 30732.00 (29793.50–33456.00) |
| all | attack | 1748 | 117.18 (109.08–126.27) | 59.60 (55.84–63.62) | 56.75 (46.09–82.19) | 125.81 (82.08–179.99) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 28303.00 (25292.00–31242.50) |

| session | n bona fide | n attack | n subjects | AUROC lum_mean [bootstrap] | AUROC lum_std [bootstrap] | AUROC sat_mean [bootstrap] | AUROC lap_var [bootstrap] | AUROC width [bootstrap] | AUROC height [bootstrap] | AUROC bytes [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 889 | 855 | 9 | 0.0779 [0.0326–0.1531] | 0.0805 [0.0000–0.2550] | 0.9573 [0.8757–0.9994] | 0.0673 [0.0000–0.2555] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.1152 [0.0068–0.3168] |
| 02 | 854 | 893 | 9 | 0.6071 [0.4695–0.7500] | 0.0394 [0.0036–0.1171] | 0.0664 [0.0189–0.1147] | 0.3765 [0.2356–0.5040] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.4439 [0.2378–0.6328] |
| all | 1743 | 1748 | 9 | 0.3334 [0.2212–0.4566] | 0.0550 [0.0020–0.1240] | 0.6136 [0.4676–0.7305] | 0.1696 [0.1010–0.2432] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.2733 [0.1841–0.3784] |

| session | subjects with both classes | within-subject AUROC lum_mean: median (min–max) | within-subject AUROC lum_std: median (min–max) | within-subject AUROC sat_mean: median (min–max) | within-subject AUROC lap_var: median (min–max) | within-subject AUROC width: median (min–max) | within-subject AUROC height: median (min–max) | within-subject AUROC bytes: median (min–max) |
|---|---|---|---|---|---|---|---|---|
| 01 | 8 | 0.0347 (0.0072–0.2687) | 0.0000 (0.0000–0.0007) | 1.0000 (0.7200–1.0000) | 0.0000 (0.0000–0.8884) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.0000 (0.0000–0.9705) |
| 02 | 7 | 0.6786 (0.2978–0.9523) | 0.0090 (0.0000–0.3756) | 0.0159 (0.0017–0.1405) | 0.4919 (0.0339–0.6648) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.6731 (0.0326–0.8794) |
| all | 9 | 0.2646 (0.0039–0.6329) | 0.0036 (0.0000–0.1249) | 0.5499 (0.0025–0.9254) | 0.1619 (0.0000–0.3851) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.2168 (0.0000–0.5654) |

### official_test (9123 images)

| session | class | n | lum_mean median (Q1–Q3) | lum_std median (Q1–Q3) | sat_mean median (Q1–Q3) | lap_var median (Q1–Q3) | width median (Q1–Q3) | height median (Q1–Q3) | bytes median (Q1–Q3) |
|---|---|---|---|---|---|---|---|---|---|
| 03 | bona fide | 3362 | 127.96 (120.37–141.73) | 84.64 (76.94–92.00) | 65.92 (60.07–72.76) | 384.36 (340.06–442.09) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 31495.00 (29910.00–32873.00) |
| 03 | attack | 5761 | 120.98 (112.07–130.54) | 68.10 (63.07–75.86) | 55.39 (50.35–60.80) | 170.79 (125.73–293.71) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 31051.00 (28009.00–34296.00) |

| session | n bona fide | n attack | n subjects | AUROC lum_mean [bootstrap] | AUROC lum_std [bootstrap] | AUROC sat_mean [bootstrap] | AUROC lap_var [bootstrap] | AUROC width [bootstrap] | AUROC height [bootstrap] | AUROC bytes [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|---|
| 03 | 3362 | 5761 | 16 | 0.3227 [0.1561–0.4920] | 0.1182 [0.0189–0.2417] | 0.2302 [0.0825–0.4146] | 0.1371 [0.0407–0.2628] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.4718 [0.2462–0.6935] |

| session | subjects with both classes | within-subject AUROC lum_mean: median (min–max) | within-subject AUROC lum_std: median (min–max) | within-subject AUROC sat_mean: median (min–max) | within-subject AUROC lap_var: median (min–max) | within-subject AUROC width: median (min–max) | within-subject AUROC height: median (min–max) | within-subject AUROC bytes: median (min–max) |
|---|---|---|---|---|---|---|---|---|
| 03 | 8 | 0.1204 (0.0062–0.7147) | 0.0003 (0.0000–0.2844) | 0.1854 (0.0034–0.6608) | 0.1553 (0.0000–0.6858) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.8756 (0.0194–0.9903) |

### cnn_train (2507 images)

| session | class | n | lum_mean median (Q1–Q3) | lum_std median (Q1–Q3) | sat_mean median (Q1–Q3) | lap_var median (Q1–Q3) | width median (Q1–Q3) | height median (Q1–Q3) | bytes median (Q1–Q3) |
|---|---|---|---|---|---|---|---|---|---|
| 01 | bona fide | 717 | 130.72 (126.83–135.20) | 82.30 (79.99–83.23) | 42.56 (38.12–46.60) | 336.00 (309.05–403.76) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 30248.00 (29638.00–32727.00) |
| 01 | attack | 568 | 112.82 (106.27–118.41) | 56.47 (52.37–60.21) | 82.42 (74.90–88.63) | 104.92 (61.20–141.93) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 27117.00 (24804.50–28680.75) |
| 02 | bona fide | 628 | 115.91 (109.55–124.35) | 74.10 (70.86–76.70) | 64.29 (59.24–67.93) | 189.95 (164.36–235.53) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 31418.00 (30356.25–34157.75) |
| 02 | attack | 594 | 123.01 (114.45–132.77) | 61.42 (58.77–63.97) | 48.69 (42.29–55.12) | 167.84 (119.45–245.85) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 31672.00 (29136.50–34137.50) |
| all | bona fide | 1345 | 126.30 (116.00–131.51) | 77.97 (74.38–82.50) | 48.77 (42.06–63.77) | 254.75 (172.83–338.50) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 30876.00 (29808.00–33258.00) |
| all | attack | 1162 | 116.87 (109.47–125.72) | 59.25 (55.68–62.77) | 58.49 (46.59–82.21) | 133.56 (85.15–192.27) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 28899.50 (26567.75–31876.50) |

| session | n bona fide | n attack | n subjects | AUROC lum_mean [bootstrap] | AUROC lum_std [bootstrap] | AUROC sat_mean [bootstrap] | AUROC lap_var [bootstrap] | AUROC width [bootstrap] | AUROC height [bootstrap] | AUROC bytes [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 717 | 568 | 7 | 0.0723 [0.0322–0.1345] | 0.0001 [0.0000–0.0002] | 0.9472 [0.8371–1.0000] | 0.0704 [0.0000–0.2723] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.1511 [0.0116–0.3806] |
| 02 | 628 | 594 | 7 | 0.6545 [0.5185–0.8001] | 0.0486 [0.0050–0.1514] | 0.0873 [0.0310–0.1376] | 0.4204 [0.2610–0.5571] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.4693 [0.2409–0.7084] |
| all | 1345 | 1162 | 7 | 0.3428 [0.2437–0.4687] | 0.0163 [0.0017–0.0510] | 0.6488 [0.5479–0.7322] | 0.1856 [0.1197–0.2571] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.3129 [0.2124–0.4298] |

| session | subjects with both classes | within-subject AUROC lum_mean: median (min–max) | within-subject AUROC lum_std: median (min–max) | within-subject AUROC sat_mean: median (min–max) | within-subject AUROC lap_var: median (min–max) | within-subject AUROC width: median (min–max) | within-subject AUROC height: median (min–max) | within-subject AUROC bytes: median (min–max) |
|---|---|---|---|---|---|---|---|---|
| 01 | 7 | 0.0382 (0.0080–0.2780) | 0.0000 (0.0000–0.0008) | 1.0000 (0.7252–1.0000) | 0.0000 (0.0000–0.8802) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.0001 (0.0000–0.9781) |
| 02 | 6 | 0.6909 (0.3005–0.9515) | 0.0112 (0.0000–0.3788) | 0.0427 (0.0132–0.1291) | 0.5326 (0.0382–0.6508) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.6966 (0.0323–0.8857) |
| all | 7 | 0.2806 (0.1179–0.6327) | 0.0042 (0.0000–0.1269) | 0.5507 (0.4449–0.9207) | 0.1705 (0.0080–0.3868) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.4152 (0.0963–0.5646) |

### cnn_es (493 images)

| session | class | n | lum_mean median (Q1–Q3) | lum_std median (Q1–Q3) | sat_mean median (Q1–Q3) | lap_var median (Q1–Q3) | width median (Q1–Q3) | height median (Q1–Q3) | bytes median (Q1–Q3) |
|---|---|---|---|---|---|---|---|---|---|
| 01 | bona fide | 49 | 136.10 (133.71–137.21) | 83.93 (83.49–84.43) | 41.95 (41.23–42.63) | 538.66 (519.91–548.58) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 37944.00 (37693.00–39843.00) |
| 01 | attack | 166 | 117.44 (104.38–126.02) | 69.19 (59.43–85.04) | 81.99 (56.81–96.61) | 106.31 (76.20–121.81) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 24060.00 (23682.00–24899.50) |
| 02 | bona fide | 106 | 132.54 (120.16–134.20) | 90.16 (89.22–90.48) | 74.94 (73.83–85.13) | 166.81 (162.88–170.54) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 29877.50 (29454.75–30100.25) |
| 02 | attack | 172 | 119.65 (111.22–134.42) | 58.11 (54.89–62.78) | 43.72 (39.39–51.04) | 126.51 (82.16–164.89) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 29327.50 (26824.00–31136.00) |
| all | bona fide | 155 | 133.54 (124.63–135.02) | 89.23 (84.43–90.34) | 73.83 (42.78–82.01) | 170.49 (164.97–515.75) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 30087.00 (29660.00–37647.00) |
| all | attack | 338 | 118.75 (107.06–127.07) | 61.31 (56.26–70.73) | 54.93 (43.67–81.31) | 112.39 (77.89–140.72) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 25483.00 (23932.25–29411.50) |

| session | n bona fide | n attack | n subjects | AUROC lum_mean [bootstrap] | AUROC lum_std [bootstrap] | AUROC sat_mean [bootstrap] | AUROC lap_var [bootstrap] | AUROC width [bootstrap] | AUROC height [bootstrap] | AUROC bytes [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 49 | 166 | 2 | 0.0108 [0.0090–0.0108] | 0.3692 [0.0000–0.3692] | 1.0000 [1.0000–1.0000] | 0.0000 [0.0000–0.0000] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.0000 [0.0000–0.0000] |
| 02 | 106 | 172 | 2 | 0.3843 [0.3843–0.6858] | 0.0000 [0.0000–0.0000] | 0.0066 [0.0020–0.0066] | 0.2594 [0.2594–0.2695] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.4501 [0.4245–0.4501] |
| all | 155 | 338 | 2 | 0.2349 [0.0045–0.5261] | 0.0697 [0.0000–0.0697] | 0.4165 [0.0030–0.9171] | 0.1149 [0.0000–0.2131] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.1566 [0.0000–0.2199] |

| session | subjects with both classes | within-subject AUROC lum_mean: median (min–max) | within-subject AUROC lum_std: median (min–max) | within-subject AUROC sat_mean: median (min–max) | within-subject AUROC lap_var: median (min–max) | within-subject AUROC width: median (min–max) | within-subject AUROC height: median (min–max) | within-subject AUROC bytes: median (min–max) |
|---|---|---|---|---|---|---|---|---|
| 01 | 1 | 0.0090 (0.0090–0.0090) | 0.0000 (0.0000–0.0000) | 1.0000 (1.0000–1.0000) | 0.0000 (0.0000–0.0000) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.0000 (0.0000–0.0000) |
| 02 | 1 | 0.6858 (0.6858–0.6858) | 0.0000 (0.0000–0.0000) | 0.0020 (0.0020–0.0020) | 0.2695 (0.2695–0.2695) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.4245 (0.4245–0.4245) |
| all | 2 | 0.2653 (0.0045–0.5261) | 0.0184 (0.0000–0.0367) | 0.4600 (0.0030–0.9171) | 0.1065 (0.0000–0.2131) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.1100 (0.0000–0.2199) |

### cnn_test (3000 images)

| session | class | n | lum_mean median (Q1–Q3) | lum_std median (Q1–Q3) | sat_mean median (Q1–Q3) | lap_var median (Q1–Q3) | width median (Q1–Q3) | height median (Q1–Q3) | bytes median (Q1–Q3) |
|---|---|---|---|---|---|---|---|---|---|
| 03 | bona fide | 1500 | 127.46 (120.35–141.87) | 84.54 (77.09–91.81) | 65.94 (60.09–72.69) | 384.39 (340.13–442.61) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 31476.00 (29901.25–32873.25) |
| 03 | attack | 1500 | 121.07 (111.65–130.26) | 68.01 (63.06–75.89) | 55.22 (50.34–60.73) | 170.02 (125.49–288.35) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 30994.50 (27901.00–34260.50) |

| session | n bona fide | n attack | n subjects | AUROC lum_mean [bootstrap] | AUROC lum_std [bootstrap] | AUROC sat_mean [bootstrap] | AUROC lap_var [bootstrap] | AUROC width [bootstrap] | AUROC height [bootstrap] | AUROC bytes [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|---|
| 03 | 1500 | 1500 | 16 | 0.3220 [0.1563–0.4925] | 0.1187 [0.0187–0.2431] | 0.2305 [0.0812–0.4159] | 0.1378 [0.0416–0.2652] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.4691 [0.2414–0.6899] |

| session | subjects with both classes | within-subject AUROC lum_mean: median (min–max) | within-subject AUROC lum_std: median (min–max) | within-subject AUROC sat_mean: median (min–max) | within-subject AUROC lap_var: median (min–max) | within-subject AUROC width: median (min–max) | within-subject AUROC height: median (min–max) | within-subject AUROC bytes: median (min–max) |
|---|---|---|---|---|---|---|---|---|
| 03 | 8 | 0.1204 (0.0011–0.7409) | 0.0001 (0.0000–0.2710) | 0.2027 (0.0011–0.6514) | 0.1686 (0.0000–0.6627) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.8520 (0.0166–0.9860) |

### c1 (300 images)

| session | class | n | lum_mean median (Q1–Q3) | lum_std median (Q1–Q3) | sat_mean median (Q1–Q3) | lap_var median (Q1–Q3) | width median (Q1–Q3) | height median (Q1–Q3) | bytes median (Q1–Q3) |
|---|---|---|---|---|---|---|---|---|---|
| 03 | bona fide | 150 | 127.22 (120.27–142.62) | 84.66 (76.32–90.56) | 65.21 (59.73–72.83) | 385.03 (337.27–440.28) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 31477.00 (30013.25–32760.25) |
| 03 | attack | 150 | 121.87 (113.59–129.78) | 67.73 (63.03–74.76) | 54.69 (49.91–59.81) | 172.63 (125.46–323.14) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 31140.50 (28120.50–34717.75) |

| session | n bona fide | n attack | n subjects | AUROC lum_mean [bootstrap] | AUROC lum_std [bootstrap] | AUROC sat_mean [bootstrap] | AUROC lap_var [bootstrap] | AUROC width [bootstrap] | AUROC height [bootstrap] | AUROC bytes [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|---|
| 03 | 150 | 150 | 16 | 0.3358 [0.1675–0.5141] | 0.1050 [0.0125–0.2205] | 0.2181 [0.0748–0.3934] | 0.1521 [0.0504–0.2945] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.4838 [0.2433–0.7101] |

| session | subjects with both classes | within-subject AUROC lum_mean: median (min–max) | within-subject AUROC lum_std: median (min–max) | within-subject AUROC sat_mean: median (min–max) | within-subject AUROC lap_var: median (min–max) | within-subject AUROC width: median (min–max) | within-subject AUROC height: median (min–max) | within-subject AUROC bytes: median (min–max) |
|---|---|---|---|---|---|---|---|---|
| 03 | 8 | 0.2200 (0.0000–0.5625) | 0.0000 (0.0000–0.2222) | 0.1364 (0.0000–0.6017) | 0.2811 (0.0000–0.6806) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.8472 (0.0000–1.0000) |

### target_491 (491 images)

| session | class | n | lum_mean median (Q1–Q3) | lum_std median (Q1–Q3) | sat_mean median (Q1–Q3) | lap_var median (Q1–Q3) | width median (Q1–Q3) | height median (Q1–Q3) | bytes median (Q1–Q3) |
|---|---|---|---|---|---|---|---|---|---|
| 01 | bona fide | 123 | 130.63 (126.62–135.02) | 82.42 (79.99–83.77) | 42.52 (39.08–46.32) | 344.25 (318.13–410.21) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 30762.00 (29765.00–34215.50) |
| 01 | attack | 121 | 111.89 (105.18–118.30) | 56.96 (52.96–63.12) | 82.67 (69.88–89.80) | 108.94 (63.19–132.96) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 26252.00 (23775.00–28364.00) |
| 02 | bona fide | 120 | 116.32 (110.36–125.88) | 75.64 (71.53–77.53) | 64.83 (60.39–74.65) | 176.85 (164.43–220.23) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 31036.00 (29934.25–32324.75) |
| 02 | attack | 127 | 121.61 (114.53–132.45) | 61.55 (57.84–63.86) | 49.15 (42.64–53.56) | 151.74 (101.54–210.56) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 30951.00 (28121.50–33364.00) |
| all | bona fide | 243 | 126.42 (116.16–131.59) | 79.50 (75.73–82.77) | 52.62 (42.50–64.66) | 252.41 (170.05–344.66) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 30990.00 (29825.50–33403.50) |
| all | attack | 248 | 116.72 (109.58–125.92) | 59.13 (55.87–63.79) | 56.24 (47.52–82.21) | 124.71 (79.69–171.91) | 640.00 (640.00–640.00) | 480.00 (480.00–480.00) | 28334.50 (25266.25–31171.25) |

| session | n bona fide | n attack | n subjects | AUROC lum_mean [bootstrap] | AUROC lum_std [bootstrap] | AUROC sat_mean [bootstrap] | AUROC lap_var [bootstrap] | AUROC width [bootstrap] | AUROC height [bootstrap] | AUROC bytes [bootstrap] |
|---|---|---|---|---|---|---|---|---|---|---|
| 01 | 123 | 121 | 9 | 0.0671 [0.0225–0.1354] | 0.0782 [0.0000–0.2408] | 0.9550 [0.8676–0.9996] | 0.0697 [0.0000–0.2636] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.1062 [0.0059–0.2983] |
| 02 | 120 | 127 | 9 | 0.6218 [0.4576–0.7780] | 0.0343 [0.0000–0.1077] | 0.0551 [0.0118–0.1120] | 0.3400 [0.1867–0.4787] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.4207 [0.2239–0.5924] |
| all | 243 | 248 | 9 | 0.3293 [0.2016–0.4552] | 0.0519 [0.0002–0.1197] | 0.6151 [0.4610–0.7414] | 0.1574 [0.0929–0.2279] | 0.5000 [0.5000–0.5000] | 0.5000 [0.5000–0.5000] | 0.2559 [0.1724–0.3491] |

| session | subjects with both classes | within-subject AUROC lum_mean: median (min–max) | within-subject AUROC lum_std: median (min–max) | within-subject AUROC sat_mean: median (min–max) | within-subject AUROC lap_var: median (min–max) | within-subject AUROC width: median (min–max) | within-subject AUROC height: median (min–max) | within-subject AUROC bytes: median (min–max) |
|---|---|---|---|---|---|---|---|---|
| 01 | 8 | 0.0227 (0.0000–0.2014) | 0.0000 (0.0000–0.0000) | 1.0000 (0.6852–1.0000) | 0.0000 (0.0000–0.9289) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.0000 (0.0000–0.9333) |
| 02 | 7 | 0.6345 (0.2863–0.9808) | 0.0000 (0.0000–0.3487) | 0.0370 (0.0000–0.2143) | 0.3532 (0.0084–0.7407) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.4365 (0.0118–0.8221) |
| all | 9 | 0.2650 (0.0000–0.6284) | 0.0000 (0.0000–0.1088) | 0.5467 (0.0000–0.9821) | 0.1796 (0.0000–0.3750) | 0.5000 (0.5000–0.5000) | 0.5000 (0.5000–0.5000) | 0.2122 (0.0000–0.5564) |

## Sets, crop view (training-cache frames, 256x256 after preprocessing)

### cnn_train (2507 frames)

| session | class | n | lum_mean median (Q1–Q3) | lum_std median (Q1–Q3) | sat_mean median (Q1–Q3) | lap_var median (Q1–Q3) |
|---|---|---|---|---|---|---|
| 01 | bona fide | 717 | 138.77 (130.21–144.47) | 69.39 (66.94–70.97) | 38.00 (35.88–40.76) | 553.02 (464.07–622.67) |
| 01 | attack | 568 | 173.33 (164.18–178.62) | 67.24 (61.90–72.55) | 48.10 (39.32–55.70) | 221.63 (171.42–279.76) |
| 02 | bona fide | 628 | 155.14 (149.73–161.85) | 68.66 (65.88–71.38) | 53.44 (49.75–58.38) | 729.94 (551.01–890.06) |
| 02 | attack | 594 | 174.04 (164.70–183.67) | 64.88 (59.36–69.34) | 27.72 (24.29–32.45) | 182.57 (106.33–279.22) |
| all | bona fide | 1345 | 146.15 (136.39–156.02) | 69.08 (66.28–71.14) | 41.73 (37.57–52.93) | 590.89 (484.08–745.52) |
| all | attack | 1162 | 173.70 (164.58–181.04) | 65.41 (60.06–71.11) | 34.52 (27.23–47.57) | 206.82 (134.26–279.64) |

| session | n bona fide | n attack | n subjects | AUROC lum_mean [bootstrap] | AUROC lum_std [bootstrap] | AUROC sat_mean [bootstrap] | AUROC lap_var [bootstrap] |
|---|---|---|---|---|---|---|---|
| 01 | 717 | 568 | 7 | 0.9815 [0.9415–0.9967] | 0.4181 [0.1335–0.7593] | 0.7622 [0.5175–0.9411] | 0.1434 [0.0001–0.3922] |
| 02 | 628 | 594 | 7 | 0.8622 [0.6191–0.9843] | 0.2981 [0.0660–0.6036] | 0.0029 [0.0000–0.0097] | 0.0191 [0.0008–0.0429] |
| all | 1345 | 1162 | 7 | 0.9221 [0.7708–0.9893] | 0.3532 [0.0934–0.6650] | 0.3028 [0.1804–0.4052] | 0.0754 [0.0049–0.1716] |

| session | subjects with both classes | within-subject AUROC lum_mean: median (min–max) | within-subject AUROC lum_std: median (min–max) | within-subject AUROC sat_mean: median (min–max) | within-subject AUROC lap_var: median (min–max) |
|---|---|---|---|---|---|
| 01 | 7 | 0.9940 (0.9291–1.0000) | 0.2766 (0.0000–0.9866) | 0.9378 (0.0130–1.0000) | 0.0000 (0.0000–0.9441) |
| 02 | 6 | 0.9815 (0.1173–0.9996) | 0.5163 (0.0004–0.9505) | 0.0000 (0.0000–0.0129) | 0.0041 (0.0000–0.0224) |
| all | 7 | 0.9873 (0.5248–0.9901) | 0.3983 (0.0002–0.9373) | 0.3466 (0.0233–0.5016) | 0.0082 (0.0000–0.3580) |

### cnn_es (493 frames)

| session | class | n | lum_mean median (Q1–Q3) | lum_std median (Q1–Q3) | sat_mean median (Q1–Q3) | lap_var median (Q1–Q3) |
|---|---|---|---|---|---|---|
| 01 | bona fide | 49 | 141.69 (139.81–144.03) | 67.56 (65.86–68.28) | 35.53 (34.55–37.38) | 713.38 (656.21–780.25) |
| 01 | attack | 166 | 172.98 (155.12–188.55) | 63.74 (62.58–64.50) | 40.88 (40.06–47.01) | 142.07 (108.29–166.76) |
| 02 | bona fide | 106 | 161.01 (155.46–163.39) | 74.15 (73.78–76.08) | 54.31 (52.69–57.83) | 469.40 (441.19–494.49) |
| 02 | attack | 172 | 173.22 (167.52–180.77) | 62.52 (60.81–64.19) | 28.16 (25.72–29.95) | 173.99 (113.90–242.14) |
| all | bona fide | 155 | 156.05 (144.20–162.83) | 73.78 (68.67–75.00) | 52.88 (39.23–55.92) | 489.96 (451.73–650.20) |
| all | attack | 338 | 173.22 (159.25–185.75) | 63.23 (61.88–64.41) | 33.62 (28.11–40.83) | 152.45 (111.93–199.62) |

| session | n bona fide | n attack | n subjects | AUROC lum_mean [bootstrap] | AUROC lum_std [bootstrap] | AUROC sat_mean [bootstrap] | AUROC lap_var [bootstrap] |
|---|---|---|---|---|---|---|---|
| 01 | 49 | 166 | 2 | 0.9785 [0.9785–1.0000] | 0.2090 [0.2090–0.2136] | 0.8413 [0.8413–0.8512] | 0.0000 [0.0000–0.0000] |
| 02 | 106 | 172 | 2 | 0.9047 [0.8613–0.9047] | 0.0000 [0.0000–0.0000] | 0.0001 [0.0000–0.0001] | 0.0093 [0.0093–0.0139] |
| all | 155 | 338 | 2 | 0.8335 [0.5535–0.9839] | 0.0613 [0.0000–0.1705] | 0.1534 [0.0000–0.4376] | 0.0033 [0.0000–0.0072] |

| session | subjects with both classes | within-subject AUROC lum_mean: median (min–max) | within-subject AUROC lum_std: median (min–max) | within-subject AUROC sat_mean: median (min–max) | within-subject AUROC lap_var: median (min–max) |
|---|---|---|---|---|---|
| 01 | 1 | 1.0000 (1.0000–1.0000) | 0.2136 (0.2136–0.2136) | 0.8512 (0.8512–0.8512) | 0.0000 (0.0000–0.0000) |
| 02 | 1 | 0.8613 (0.8613–0.8613) | 0.0000 (0.0000–0.0000) | 0.0000 (0.0000–0.0000) | 0.0139 (0.0139–0.0139) |
| all | 2 | 0.7687 (0.5535–0.9839) | 0.0853 (0.0000–0.1705) | 0.2188 (0.0000–0.4376) | 0.0036 (0.0000–0.0072) |

### cnn_test (3000 frames)

| session | class | n | lum_mean median (Q1–Q3) | lum_std median (Q1–Q3) | sat_mean median (Q1–Q3) | lap_var median (Q1–Q3) |
|---|---|---|---|---|---|---|
| 03 | bona fide | 1500 | 166.87 (155.02–173.97) | 73.72 (70.37–77.24) | 45.12 (37.93–57.62) | 846.47 (674.28–986.40) |
| 03 | attack | 1500 | 157.67 (147.86–165.59) | 66.38 (63.03–70.35) | 43.33 (39.12–47.47) | 288.65 (210.23–443.15) |

| session | n bona fide | n attack | n subjects | AUROC lum_mean [bootstrap] | AUROC lum_std [bootstrap] | AUROC sat_mean [bootstrap] | AUROC lap_var [bootstrap] |
|---|---|---|---|---|---|---|---|
| 03 | 1500 | 1500 | 16 | 0.3531 [0.1416–0.6001] | 0.2133 [0.0694–0.3956] | 0.4014 [0.1352–0.6686] | 0.0678 [0.0128–0.1504] |

| session | subjects with both classes | within-subject AUROC lum_mean: median (min–max) | within-subject AUROC lum_std: median (min–max) | within-subject AUROC sat_mean: median (min–max) | within-subject AUROC lap_var: median (min–max) |
|---|---|---|---|---|---|
| 03 | 8 | 0.2726 (0.0000–0.9939) | 0.5532 (0.0000–0.9916) | 0.4137 (0.0000–0.9818) | 0.0784 (0.0000–0.5570) |

### c1 (300 frames)

| session | class | n | lum_mean median (Q1–Q3) | lum_std median (Q1–Q3) | sat_mean median (Q1–Q3) | lap_var median (Q1–Q3) |
|---|---|---|---|---|---|---|
| 03 | bona fide | 150 | 167.12 (155.00–173.57) | 74.03 (71.12–78.05) | 44.92 (37.52–57.42) | 841.43 (668.68–986.14) |
| 03 | attack | 150 | 159.37 (147.71–165.42) | 66.69 (63.19–70.86) | 43.77 (38.88–47.29) | 277.47 (221.41–484.94) |

| session | n bona fide | n attack | n subjects | AUROC lum_mean [bootstrap] | AUROC lum_std [bootstrap] | AUROC sat_mean [bootstrap] | AUROC lap_var [bootstrap] |
|---|---|---|---|---|---|---|---|
| 03 | 150 | 150 | 16 | 0.3744 [0.1530–0.6258] | 0.2110 [0.0636–0.3968] | 0.4041 [0.1312–0.6794] | 0.0849 [0.0211–0.1869] |

| session | subjects with both classes | within-subject AUROC lum_mean: median (min–max) | within-subject AUROC lum_std: median (min–max) | within-subject AUROC sat_mean: median (min–max) | within-subject AUROC lap_var: median (min–max) |
|---|---|---|---|---|---|
| 03 | 8 | 0.2818 (0.0000–1.0000) | 0.5383 (0.0000–1.0000) | 0.3685 (0.0000–0.9935) | 0.0764 (0.0000–0.6753) |

## Spearman between each statistic and score_attack (per score file, session and class)

### N1 LivenessNet seed 42, session 03 (cnn_test 3,000), run 20261024-N1-LivenessNet-fb93496 — raw view (cnn_test, 3000 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var | width | height | bytes |
|---|---|---|---|---|---|---|---|---|---|
| 03 | bona fide | 1500 | 0.2834 | 0.3937 | -0.0914 | 0.1405 | NA | NA | -0.4558 |
| 03 | attack | 1500 | -0.2013 | -0.0924 | 0.0611 | 0.5375 | NA | NA | 0.5990 |
| 03 | all | 3000 | 0.1106 | 0.3057 | 0.1308 | 0.3643 | NA | NA | 0.0980 |

### N1 LivenessNet seed 42, session 03 (cnn_test 3,000), run 20261024-N1-LivenessNet-fb93496 — crop view (cnn_test, 3000 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var |
|---|---|---|---|---|---|---|
| 03 | bona fide | 1500 | 0.8139 | 0.7016 | -0.8348 | -0.6448 |
| 03 | attack | 1500 | 0.2305 | 0.4313 | -0.0286 | 0.3301 |
| 03 | all | 3000 | 0.6344 | 0.6705 | -0.4912 | 0.1004 |

### N1 LivenessNet seed 43, session 03 (cnn_test 3,000), run 20261024-N1-LivenessNet-fb93496 — raw view (cnn_test, 3000 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var | width | height | bytes |
|---|---|---|---|---|---|---|---|---|---|
| 03 | bona fide | 1500 | 0.3808 | 0.2996 | -0.1844 | 0.0811 | NA | NA | -0.4257 |
| 03 | attack | 1500 | -0.2061 | -0.1402 | 0.0718 | 0.5201 | NA | NA | 0.5970 |
| 03 | all | 3000 | 0.1511 | 0.2625 | 0.0830 | 0.3389 | NA | NA | 0.0852 |

### N1 LivenessNet seed 43, session 03 (cnn_test 3,000), run 20261024-N1-LivenessNet-fb93496 — crop view (cnn_test, 3000 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var |
|---|---|---|---|---|---|---|
| 03 | bona fide | 1500 | 0.8705 | 0.5743 | -0.7442 | -0.6240 |
| 03 | attack | 1500 | 0.3862 | 0.3151 | -0.1128 | 0.3133 |
| 03 | all | 3000 | 0.7005 | 0.5853 | -0.5374 | 0.0532 |

### N1 LivenessNet seed 44, session 03 (cnn_test 3,000), run 20261024-N1-LivenessNet-fb93496 — raw view (cnn_test, 3000 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var | width | height | bytes |
|---|---|---|---|---|---|---|---|---|---|
| 03 | bona fide | 1500 | 0.2825 | 0.2830 | -0.2333 | -0.0652 | NA | NA | -0.4107 |
| 03 | attack | 1500 | -0.1850 | -0.1909 | 0.0654 | 0.3752 | NA | NA | 0.4757 |
| 03 | all | 3000 | 0.0884 | 0.1463 | -0.0053 | 0.1786 | NA | NA | 0.0710 |

### N1 LivenessNet seed 44, session 03 (cnn_test 3,000), run 20261024-N1-LivenessNet-fb93496 — crop view (cnn_test, 3000 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var |
|---|---|---|---|---|---|---|
| 03 | bona fide | 1500 | 0.7246 | 0.4414 | -0.6360 | -0.6971 |
| 03 | attack | 1500 | 0.3907 | 0.1381 | -0.1464 | 0.1354 |
| 03 | all | 3000 | 0.6171 | 0.3776 | -0.4688 | -0.1250 |

### N2 part 1, N1 checkpoint seed 42, sessions 01-02 (491 target frames), study 20261029-N2-sessions-2ac3e29 — raw view (target_491, 491 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var | width | height | bytes |
|---|---|---|---|---|---|---|---|---|---|
| 01 | bona fide | 123 | -0.0585 | -0.4129 | -0.3053 | -0.4649 | NA | NA | -0.6119 |
| 01 | attack | 121 | -0.2237 | -0.2675 | 0.1967 | -0.3111 | NA | NA | -0.0012 |
| 01 | all | 244 | -0.7102 | -0.7840 | 0.6771 | -0.8055 | NA | NA | -0.7406 |
| 02 | bona fide | 120 | 0.3021 | 0.3514 | 0.0543 | -0.1787 | NA | NA | -0.2681 |
| 02 | attack | 127 | 0.0246 | 0.0294 | 0.0310 | -0.1554 | NA | NA | -0.1206 |
| 02 | all | 247 | 0.2666 | -0.6772 | -0.6731 | -0.3187 | NA | NA | -0.2104 |
| all | bona fide | 243 | -0.0272 | -0.0523 | 0.2132 | -0.5808 | NA | NA | -0.4467 |
| all | attack | 248 | -0.1184 | -0.1563 | 0.0958 | -0.2099 | NA | NA | -0.0512 |
| all | all | 491 | -0.2934 | -0.7394 | 0.2567 | -0.6745 | NA | NA | -0.4837 |

### N2 part 1, N1 checkpoint seed 42, sessions 01-02 (491 target frames), study 20261029-N2-sessions-2ac3e29 — crop view: NA: frames not in the training cache

### N2 part 1, N1 checkpoint seed 43, sessions 01-02 (491 target frames), study 20261029-N2-sessions-2ac3e29 — raw view (target_491, 491 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var | width | height | bytes |
|---|---|---|---|---|---|---|---|---|---|
| 01 | bona fide | 123 | -0.3731 | -0.4998 | -0.0313 | -0.1056 | NA | NA | -0.2333 |
| 01 | attack | 121 | -0.4148 | -0.0801 | 0.5510 | -0.1484 | NA | NA | 0.0510 |
| 01 | all | 244 | -0.7760 | -0.7626 | 0.7604 | -0.6855 | NA | NA | -0.6118 |
| 02 | bona fide | 120 | 0.0948 | 0.1774 | 0.0957 | 0.0919 | NA | NA | -0.0463 |
| 02 | attack | 127 | -0.2316 | -0.1512 | 0.3841 | -0.2043 | NA | NA | -0.1090 |
| 02 | all | 247 | 0.1476 | -0.7064 | -0.5914 | -0.2732 | NA | NA | -0.1643 |
| all | bona fide | 243 | 0.2129 | 0.3786 | -0.2586 | 0.2158 | NA | NA | -0.1703 |
| all | attack | 248 | -0.2969 | -0.1290 | 0.3284 | -0.1776 | NA | NA | -0.0279 |
| all | all | 491 | -0.2842 | -0.6697 | 0.1965 | -0.5115 | NA | NA | -0.4099 |

### N2 part 1, N1 checkpoint seed 43, sessions 01-02 (491 target frames), study 20261029-N2-sessions-2ac3e29 — crop view: NA: frames not in the training cache

### N2 part 1, N1 checkpoint seed 44, sessions 01-02 (491 target frames), study 20261029-N2-sessions-2ac3e29 — raw view (target_491, 491 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var | width | height | bytes |
|---|---|---|---|---|---|---|---|---|---|
| 01 | bona fide | 123 | -0.1958 | -0.1812 | 0.1569 | 0.1196 | NA | NA | -0.0043 |
| 01 | attack | 121 | -0.4387 | -0.1116 | 0.5078 | -0.2312 | NA | NA | -0.0153 |
| 01 | all | 244 | -0.7600 | -0.7242 | 0.7788 | -0.6936 | NA | NA | -0.6120 |
| 02 | bona fide | 120 | 0.1001 | 0.1168 | 0.0845 | 0.1262 | NA | NA | 0.0416 |
| 02 | attack | 127 | -0.1474 | -0.0726 | 0.2355 | 0.1277 | NA | NA | 0.1230 |
| 02 | all | 247 | 0.1708 | -0.6999 | -0.6232 | -0.1806 | NA | NA | -0.0795 |
| all | bona fide | 243 | -0.0943 | -0.0189 | 0.2378 | -0.1450 | NA | NA | 0.0667 |
| all | attack | 248 | -0.2493 | -0.1060 | 0.2109 | -0.0213 | NA | NA | 0.0790 |
| all | all | 491 | -0.3388 | -0.7207 | 0.2780 | -0.5438 | NA | NA | -0.3371 |

### N2 part 1, N1 checkpoint seed 44, sessions 01-02 (491 target frames), study 20261029-N2-sessions-2ac3e29 — crop view: NA: frames not in the training cache

### N2 part 2, combined AttackNet V2.2, whole image, session 03 (cnn_test 3,000), study 20261029-N2-combined-247dfd7 — raw view (cnn_test, 3000 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var | width | height | bytes |
|---|---|---|---|---|---|---|---|---|---|
| 03 | bona fide | 1500 | -0.2466 | 0.0466 | 0.0966 | 0.0379 | NA | NA | 0.0269 |
| 03 | attack | 1500 | -0.1843 | 0.2990 | 0.2473 | 0.3158 | NA | NA | 0.2269 |
| 03 | all | 3000 | -0.0232 | 0.4598 | 0.4172 | 0.4609 | NA | NA | 0.1500 |

### N2 part 2, combined AttackNet V2.2, whole image, session 03 (cnn_test 3,000), study 20261029-N2-combined-247dfd7 — crop view (cnn_test, 3000 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var |
|---|---|---|---|---|---|---|
| 03 | bona fide | 1500 | -0.1671 | -0.0333 | -0.0079 | 0.1492 |
| 03 | attack | 1500 | 0.1262 | 0.2506 | 0.1989 | 0.3157 |
| 03 | all | 3000 | 0.1394 | 0.3034 | 0.1341 | 0.4787 |

### N2 part 2, combined AttackNet V2.2, face crop, session 03 (cnn_test 3,000), study 20261029-N2-combined-247dfd7 — raw view (cnn_test, 3000 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var | width | height | bytes |
|---|---|---|---|---|---|---|---|---|---|
| 03 | bona fide | 1500 | 0.0388 | 0.2915 | 0.3070 | 0.2358 | NA | NA | -0.3844 |
| 03 | attack | 1500 | 0.0014 | -0.0792 | 0.0156 | 0.0504 | NA | NA | 0.1075 |
| 03 | all | 3000 | 0.0768 | 0.2226 | 0.2442 | 0.1902 | NA | NA | -0.1177 |

### N2 part 2, combined AttackNet V2.2, face crop, session 03 (cnn_test 3,000), study 20261029-N2-combined-247dfd7 — crop view (cnn_test, 3000 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var |
|---|---|---|---|---|---|---|
| 03 | bona fide | 1500 | 0.4331 | 0.7829 | -0.7012 | -0.3644 |
| 03 | attack | 1500 | -0.0685 | 0.1061 | -0.0817 | -0.0158 |
| 03 | all | 3000 | 0.3139 | 0.5362 | -0.4187 | 0.0420 |

### C1 LivenessNet (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9 — raw view (c1, 300 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var | width | height | bytes |
|---|---|---|---|---|---|---|---|---|---|
| 03 | bona fide | 150 | 0.2713 | 0.1584 | -0.0943 | -0.0363 | NA | NA | -0.2732 |
| 03 | attack | 150 | -0.1982 | -0.0911 | -0.0209 | 0.4593 | NA | NA | 0.5113 |
| 03 | all | 300 | 0.1257 | 0.2262 | 0.0613 | 0.2753 | NA | NA | 0.1135 |

### C1 LivenessNet (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9 — crop view (c1, 300 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var |
|---|---|---|---|---|---|---|
| 03 | bona fide | 150 | 0.5900 | 0.5673 | -0.6555 | -0.7050 |
| 03 | attack | 150 | 0.2285 | 0.3196 | 0.0066 | 0.2115 |
| 03 | all | 300 | 0.5755 | 0.5416 | -0.4433 | -0.0074 |

### C1 AttackNet V1 (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9 — raw view (c1, 300 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var | width | height | bytes |
|---|---|---|---|---|---|---|---|---|---|
| 03 | bona fide | 150 | 0.2622 | 0.0142 | 0.2131 | 0.0301 | NA | NA | -0.1703 |
| 03 | attack | 150 | 0.1769 | -0.1296 | -0.0389 | -0.0879 | NA | NA | -0.0032 |
| 03 | all | 300 | 0.1145 | -0.3067 | -0.1304 | -0.2818 | NA | NA | -0.0447 |

### C1 AttackNet V1 (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9 — crop view (c1, 300 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var |
|---|---|---|---|---|---|---|
| 03 | bona fide | 150 | 0.5632 | 0.5885 | -0.5915 | -0.3875 |
| 03 | attack | 150 | 0.2638 | -0.3357 | -0.5741 | -0.1803 |
| 03 | all | 300 | 0.2761 | -0.0781 | -0.5827 | -0.4752 |

### C1 AttackNet V2.1 (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9 — raw view (c1, 300 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var | width | height | bytes |
|---|---|---|---|---|---|---|---|---|---|
| 03 | bona fide | 150 | 0.4798 | 0.3421 | 0.0827 | 0.2808 | NA | NA | -0.3033 |
| 03 | attack | 150 | -0.0122 | -0.1567 | -0.0549 | 0.0662 | NA | NA | 0.1714 |
| 03 | all | 300 | 0.1431 | -0.1697 | -0.1833 | -0.1116 | NA | NA | 0.0025 |

### C1 AttackNet V2.1 (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9 — crop view (c1, 300 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var |
|---|---|---|---|---|---|---|
| 03 | bona fide | 150 | 0.7532 | 0.6917 | -0.8231 | -0.6233 |
| 03 | attack | 150 | -0.0031 | 0.0359 | -0.2206 | -0.0160 |
| 03 | all | 300 | 0.3076 | 0.1250 | -0.6247 | -0.4295 |

### C1 AttackNet V2.2 (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9 — raw view (c1, 300 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var | width | height | bytes |
|---|---|---|---|---|---|---|---|---|---|
| 03 | bona fide | 150 | 0.4076 | 0.2649 | 0.0592 | 0.2106 | NA | NA | -0.2691 |
| 03 | attack | 150 | 0.0536 | -0.1666 | -0.0648 | 0.3082 | NA | NA | 0.4445 |
| 03 | all | 300 | 0.2257 | 0.0826 | 0.0173 | 0.1880 | NA | NA | 0.1389 |

### C1 AttackNet V2.2 (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9 — crop view (c1, 300 frames)

| session | class | n | lum_mean | lum_std | sat_mean | lap_var |
|---|---|---|---|---|---|---|
| 03 | bona fide | 150 | 0.8235 | 0.7359 | -0.8512 | -0.5935 |
| 03 | attack | 150 | 0.3585 | 0.1001 | -0.2398 | 0.1497 |
| 03 | all | 300 | 0.6171 | 0.4080 | -0.6308 | -0.1226 |

## Bona fide of session 03: rejected against accepted (AUROC of each statistic, rejected = positive)

- N1 LivenessNet seed 42, session 03 (cnn_test 3,000), run 20261024-N1-LivenessNet-fb93496, raw view: rejected 1155, accepted 345; lum_mean 0.5024, lum_std 0.5363, sat_mean 0.4978, lap_var 0.3862, width 0.5000, height 0.5000, bytes 0.2276
- N1 LivenessNet seed 42, session 03 (cnn_test 3,000), run 20261024-N1-LivenessNet-fb93496, crop view: rejected 1155, accepted 345; lum_mean 0.9811, lum_std 0.9437, sat_mean 0.0184, lap_var 0.1809
- N1 LivenessNet seed 43, session 03 (cnn_test 3,000), run 20261024-N1-LivenessNet-fb93496, raw view: rejected 1152, accepted 348; lum_mean 0.5034, lum_std 0.5420, sat_mean 0.4979, lap_var 0.3902, width 0.5000, height 0.5000, bytes 0.2256
- N1 LivenessNet seed 43, session 03 (cnn_test 3,000), run 20261024-N1-LivenessNet-fb93496, crop view: rejected 1152, accepted 348; lum_mean 0.9825, lum_std 0.9453, sat_mean 0.0178, lap_var 0.1818
- N1 LivenessNet seed 44, session 03 (cnn_test 3,000), run 20261024-N1-LivenessNet-fb93496, raw view: rejected 1131, accepted 369; lum_mean 0.5178, lum_std 0.5729, sat_mean 0.4726, lap_var 0.4139, width 0.5000, height 0.5000, bytes 0.2498
- N1 LivenessNet seed 44, session 03 (cnn_test 3,000), run 20261024-N1-LivenessNet-fb93496, crop view: rejected 1131, accepted 369; lum_mean 0.9781, lum_std 0.9230, sat_mean 0.0086, lap_var 0.1606
- C1 LivenessNet (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9, raw view: rejected 109, accepted 41; lum_mean 0.5549, lum_std 0.6319, sat_mean 0.4133, lap_var 0.4408, width 0.5000, height 0.5000, bytes 0.2887
- C1 LivenessNet (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9, crop view: rejected 109, accepted 41; lum_mean 0.9705, lum_std 0.8906, sat_mean 0.0219, lap_var 0.0924
- C1 AttackNet V1 (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9, raw view: rejected 40, accepted 110; lum_mean 0.7325, lum_std 0.4725, sat_mean 0.6625, lap_var 0.5457, width 0.5000, height 0.5000, bytes 0.4665
- C1 AttackNet V1 (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9, crop view: rejected 40, accepted 110; lum_mean 0.7211, lum_std 0.7677, sat_mean 0.2441, lap_var 0.2893
- C1 AttackNet V2.1 (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9, raw view: rejected 115, accepted 35; lum_mean 0.5799, lum_std 0.6728, sat_mean 0.5702, lap_var 0.5471, width 0.5000, height 0.5000, bytes 0.2472
- C1 AttackNet V2.1 (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9, crop view: rejected 115, accepted 35; lum_mean 0.8981, lum_std 0.8979, sat_mean 0.0512, lap_var 0.2022
- C1 AttackNet V2.2 (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9, raw view: rejected 118, accepted 32; lum_mean 0.5169, lum_std 0.5228, sat_mean 0.5599, lap_var 0.3909, width 0.5000, height 0.5000, bytes 0.2677
- C1 AttackNet V2.2 (NUAA checkpoint of 2026-09-12), session 03 (c1 300), run 20260928-C1-seed42-a80f6f9, crop view: rejected 118, accepted 32; lum_mean 0.9788, lum_std 0.9251, sat_mean 0.0649, lap_var 0.2042

## Face-box annex (c1; relative area = bbox_w * bbox_h / (width * height) of the original image)

| extractor | n bona fide | n attack | bona fide median (Q1–Q3) | attack median (Q1–Q3) | class AUROC |
|---|---|---|---|---|---|
| cache_0ac7143 | 150 | 150 | 0.0854 (0.0666–0.1036) | 0.1078 (0.0829–0.1313) | 0.6975 |
| c1_a80f6f9 | 150 | 150 | 0.0844 (0.0656–0.1024) | 0.1066 (0.0819–0.1300) | 0.6975 |

Figure: figure_distributions.png: boxplots of the four raw statistics per session and class on the official lists.

Limits: Frames of one subject and session are correlated: n counts frames, not independent units. Descriptive measures without statistical tests; the only uncertainty is the bootstrap over subjects of the class AUROC. The aggregate class AUROC and the within-class Spearman confound subject and class: in session 03 the bona fide frames come from 9 subjects and the attacks from 15 subjects, 8 subjects have both classes; the within-subject AUROC is the reading that holds the subject fixed. The statistics are global image properties; they do not identify what a network uses. The only face-level view is the crop view; the raw view is the whole frame (background included; NUAA attacks are re-photographed prints with a border). The crop view describes the network input after the reference enhancement chain (bilateral filter, CLAHE, unsharp mask, gamma and a final linear contrast/brightness step), not the face region of the original. In the all row of a set with two sessions the sessions of a subject are pooled, for the within-subject AUROC and for the bootstrap alike. The crop view covers the 6,000 cache frames only (cnn_train, cnn_es, cnn_test); the 491 target frames have the raw view only. Face boxes exist only for the 300 C1 images (two extractors); no boxes for the 3,000 without a new detection pass, which is outside this step. Associations, not causes.
