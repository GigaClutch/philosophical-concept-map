import wikipedia

def test_wiki():
    print("Testing Wikipedia API...")
    
    test_concepts = ["Ethics", "Philosophy", "Justice", "Plato"]
    
    for concept in test_concepts:
        print(f"\nTrying to fetch concept: '{concept}'")
        try:
            page = wikipedia.page(concept, auto_suggest=True)
            print(f"Success! Page title: {page.title}")
            print(f"Content length: {len(page.content)} characters")
        except Exception as e:
            print(f"Error accessing Wikipedia for '{concept}': {e}")
            
        # Try without auto_suggest
        try:
            print(f"Trying again without auto_suggest...")
            page = wikipedia.page(concept, auto_suggest=False)
            print(f"Success! Page title: {page.title}")
        except Exception as e:
            print(f"Error without auto_suggest: {e}")
            
        # Try with disambiguation disabled
        try:
            print(f"Trying with disambiguation disabled...")
            page = wikipedia.page(concept, auto_suggest=True, redirect=True, preload=True)
            print(f"Success! Page title: {page.title}")
        except Exception as e:
            print(f"Error with disambiguation disabled: {e}")

if __name__ == "__main__":
    test_wiki()