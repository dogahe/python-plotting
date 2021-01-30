import numpy as np  
x_values = np.linspace(-2.25,2.25,300)  
#y_values = np.array([np.sin(x) + np.random.randn()*.25 for x in x_values])
y_values = np.array([np.sin(x) for x in x_values])

from sklearn.linear_model import LinearRegression  
lm = LinearRegression().fit(x_values.reshape(-1,1), y_values)

print("Prediction for input = 5")
print(lm.predict(5.0))


from coremltools.converters import sklearn  
coreml_model = sklearn.convert(lm)  
print(type(coreml_model))

coreml_model.author = "DSX"
coreml_model.short_description = "I approximate a sine curve with a linear model!"  
coreml_model.input_description["input"] = "a real number"  
coreml_model.output_description["prediction"] = "a real number"
print(coreml_model.author)

coreml_model.save('linear_model.mlmodel') 
