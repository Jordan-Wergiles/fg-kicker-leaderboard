
import React, { useState } from 'react';

function App() {
  const [response, setResponse] = useState(null);

  const handlePredict = async () => {
    const res = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ distance: 45, makes_in_a_row: 3 })
    });
    const data = await res.json();
    setResponse(data);
  };

  return (
    <div>
      <h1>FGOE Predictor</h1>
      <button onClick={handlePredict}>Predict</button>
      {response && <pre>{JSON.stringify(response, null, 2)}</pre>}
    </div>
  );
}

export default App;
