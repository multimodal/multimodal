import re

punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]
commaStrip = re.compile(r"(\d)(\,)(\d)")
periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")


def processPunctuation(inText):
    outText = inText
    if re.search(commaStrip, inText) != None:
        outText.replace(",", "")

    for p in punct:
        if p + " " in inText or " " + p in inText:
            outText = outText.replace(p, "")
        elif p in inText:
            outText = outText.replace(p, " ")
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText
