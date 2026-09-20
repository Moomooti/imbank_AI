"""FLORES-200 / NLLB language codes for the FinHOLLY target set.

대상 8종: 베트남 · 인니 · 태국 · 크메르 · 미얀마 · 필리핀 · 네팔 · (우즈베크=보류)
"""

SOURCE_LANG = "kor_Hang"       

TARGET_LANGS = {
    "vi": "vie_Latn",        
    "id": "ind_Latn",          
    "th": "tha_Thai",       
    "km": "khm_Khmr",        
    "my": "mya_Mymr",        
    "tl": "tgl_Latn",              
    "ne": "npi_Deva",       
                                                     
}

                                                              
                                            
                    
                             
GROUP_A = ["vi", "id", "th", "km", "my"]
GROUP_D = ["tl"]
GROUP_BC = ["ne"]

                                                    
                                                                
                                            
                                                              
GLOSSARY_LANGS = {
    "vie_Latn": {"flores": "vie_Latn", "deepl": "VI"},
    "ind_Latn": {"flores": "ind_Latn", "deepl": "ID"},
    "tha_Thai": {"flores": "tha_Thai", "deepl": "TH"},
    "tgl_Latn": {"flores": "tgl_Latn", "deepl": None},                               
    "mya_Mymr": {"flores": "mya_Mymr", "deepl": None},                                   
}
