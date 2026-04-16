Remove-Item eval_results\ts_lstm.txt -ErrorAction SilentlyContinue

python ..\test-suite-sql-eval\evaluation.py `
    --gold predictions\gold.txt `
    --pred predictions\pred_lstm.txt `
    --db data\testsuitedatabases\database `
    --table spider_data\spider_data\tables.json `
    --etype all `
    --plug_value *> eval_results\ts_lstm.txt

Write-Host "TS run done. Last 60 lines:"
Get-Content eval_results\ts_lstm.txt -Tail 60