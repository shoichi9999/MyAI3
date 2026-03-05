# CLAUDE.md

## 開発ルール

### grid_search.py の実行
- grid_search.py は長時間かかるため、バックグラウンド実行すること
- **方法1: Bashツールの `run_in_background`** （推奨）
  - 完了通知が自動で届くので便利
- **方法2: `setsid` + `nohup` + `disown`**
  - Bashツールのプロセスグループごとkillされないよう、`setsid` で新セッションを作り `disown` で切り離す
  - 実行コマンド例:
    ```bash
    setsid nohup python grid_search.py --race derby > logs/grid_derby.log 2>&1 &
    disown
    ```
  - ログ確認: `tail -f logs/grid_derby.log`

### grid_search.py 実行時の注意
- **パイプ (`| head` 等) を絶対に使わない**: パイプ先が閉じるとSIGPIPEでプロセスが即死する
  - NG: `python grid_search.py --race oaks 2>&1 | head -40`
- 方法2の場合、**`setsid` + `disown` を必ず使う**: Bashツールはタイムアウト時にプロセスグループごとSIGKILLするため、`nohup` だけでは不十分
  - NG: `nohup python grid_search.py ... &`（同じプロセスグループなのでkillされる）
  - OK: `setsid nohup python grid_search.py ... & disown`（新セッション＋切り離し）
