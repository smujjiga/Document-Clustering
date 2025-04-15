from sklearn.datasets import fetch_20newsgroups
import pandas as pd

def load_20newsgroups(categories:list|None=None, clean:bool=False, limit_to:int=1000, random_state:int=3):
    remove=("headers", "footers", "quotes") if clean else ()
    dataset = fetch_20newsgroups(    
        remove=remove,
        subset="test",
        categories=categories,
        shuffle=True,        
        random_state=random_state,
    )
    
    df = pd.DataFrame({
        "document": dataset.data,
        "label": dataset.target
    })

    return df