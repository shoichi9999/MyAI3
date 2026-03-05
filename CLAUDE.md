# CLAUDE.md

## 開発ルール

### grid_search.py の実行
- grid_search.py は長時間かかるため、Bashツールの `run_in_background` でバックグラウンド実行すること
- **パイプ (`| head` 等) を絶対に使わない**: パイプ先が閉じるとSIGPIPEでプロセスが即死する
