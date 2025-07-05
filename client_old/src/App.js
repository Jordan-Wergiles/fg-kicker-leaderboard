import React, { useState } from "react";

function App() {
  const [distance, setDistance] = useState(45);
  const [makesInARow, setMakesInARow] = useState(3);
  const [response, setResponse] = useState(null);

  const handlePredict = async () => {
    const res = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        attempt_yards: distance,
        makes_in_a_row: makesInARow
      }),
    });
    const data = await res.json();
    setResponse(data);
  };

  return (
    <div style={{ padding: "2rem", fontFamily: "Arial" }}>
      <h1>FGOE Predictor</h1>
      <label>
        Distance (yards):
        <input
          type="number"
          value={distance}
          onChange={(e) => setDistance(parseInt(e.target.value))}
        />
      </label>
      <br />
      <label>
        Makes In A Row:
        <input
          type="number"
          value={makesInARow}
          onChange={(e) => setMakesInARow(parseInt(e.target.value))}
        />
      </label>
      <br />
      <button onClick={handlePredict}>Predict</button>
      {response && (
        <div>
          <h3>Prediction</h3>
          <pre>{JSON.stringify(response, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;