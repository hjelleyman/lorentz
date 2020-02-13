from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    
def splitdf(df, n = 3):
    split = [df.iloc[i*(len(df)//n):(i+1)*(len(df)//n)] for i in range(n-1)] + [df.iloc[(n-1)*(len(df)//n):]]
    display_side_by_side(*split)
    