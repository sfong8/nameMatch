import fuzzyset

fz = fuzzyset.FuzzySet()
#Create a list of terms we would like to match against in a fuzzy way
for l in ["Diane Abbott", "Boris Johnson"]:
    fz.add(l)

#Now see if our sample term fuzzy matches any of those specified terms
sample_term='Boris Johnstone'
fz.get(sample_term)
#   , fz.get('Diana Abbot'), fz.get('Joanna Lumley')