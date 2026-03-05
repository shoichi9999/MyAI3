# CLAUDE.md

## 開発ルール

### grid_search.py の実行
- grid_search.py は長時間かかるため、Bashツールの `run_in_background` でバックグラウンド実行すること
- **パイプ (`| head` 等) を絶対に使わない**: パイプ先が閉じるとSIGPIPEでプロセスが即死する
- バックグラウンド実行後は `sleep` + `tail` で定期的にログを監視し、完了まで追跡すること
- 各Phase/DE完了ごとにチェックポイントが保存される。異常終了時は `--resume` で再開可能
