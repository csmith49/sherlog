import altair as alt
import pandas as pd

print("Loading data...")
source = pd.read_json("grad-variance-results.jsonl", lines=True)

def ci(source, x, y):
    line = alt.Chart(source).mark_line().encode(x=x, y=f"mean({y})")
    band = alt.Chart(source).mark_errorband(extent="ci").encode(x=x, y=y)
    return line + band

print("Generating charts...")
likelihood = ci(source, "step", "likelihood")
p = ci(source, "step", "p")
q = ci(source, "step", "q")

chart = (p | q) & likelihood

print("Saving to grad-variance-results.html...")
chart.save("grad-variance-results.html")
print("Done.")