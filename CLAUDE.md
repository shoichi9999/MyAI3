# CLAUDE.md

## 開発ルール

### grid_search.py の実行
- grid_search.py は長時間かかるため、nohupでバックグラウンド実行すること
- セッション切断時にプロセスが死なないよう、必ずnohupを使う
- 実行コマンド例:
  ```bash
  nohup python grid_search.py --race derby > logs/grid_derby.log 2>&1 &
  nohup python grid_search.py --race oaks > logs/grid_oaks.log 2>&1 &
  ```
- ログ確認: `tail -f logs/grid_derby.log`
