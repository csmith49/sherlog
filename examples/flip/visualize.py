import altair
import tablib

df = tablib.Dataset().load("flip-results.jsonl").export("df")