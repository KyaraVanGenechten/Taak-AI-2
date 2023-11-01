import streamlit as st
from simpleai.search import CspProblem, backtrack

# Title
st.title("Cryptarithmetic Puzzles")


word1 = st.text_input("Please enter the first word of the cryptarithmetic puzzle:", "").replace(' ', '').upper()
word2 = st.text_input("Please enter the second word of the cryptarithmetic puzzle: ").replace(' ', '').upper()
result = st.text_input("Please enter the result of the cryptarithmetic puzzle: ").replace(' ', '').upper()
st.write(f"  {word1}")
st.write(f"<b>+</b> {word2}", unsafe_allow_html=True)
st.write(f"= {result}" )

if st.button("Solve"):
    variables = set(word1 + word2 + result)
    # Rest of your code...
    domains = {letter: list(range(1, 10)) if letter in [word1[0], word2[0], result[0]] else list(range(10)) for letter in variables}


    def constraint_unique(variables, values):
        return len(values) == len(set(values)) 

    def constraint_add(variables, values):
        factor1 = int(''.join(str(values[variables.index(letter)]) for letter in word1))
        factor2 = int(''.join(str(values[variables.index(letter)]) for letter in word2))
        result_value = int(''.join(str(values[variables.index(letter)]) for letter in result))
        return (factor1 + factor2) == result_value

    constraints = [
        (variables, constraint_unique),
        (variables, constraint_add),
    ]


    problem = CspProblem(variables, domains, constraints)
    output = backtrack(problem)


    if output:
        for letter in output:
            st.write(f"{letter} = {output[letter]}")
    else:    
        st.write("No solution found")
