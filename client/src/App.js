// ... imports ...
import React, { useState, useEffect } from 'react';
import Papa from "papaparse";
import broncosLogo from './broncos_logo.png';
import broncosPrimaryLogo from './Broncos_Primary_Logo.png';
import './App.css';
import {
  BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

// App Component Initialization
function App() {
  const [showApp, setShowApp] = useState(false);
  const [activeTab, setActiveTab] = useState("predictor");
  const [response, setResponse] = useState(null);

  const [leaderboardData, setLeaderboardData] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'ascending' });
  const [currentPage, setCurrentPage] = useState(1);
  const playersPerPage = 10;

  const [seasonFilter, setSeasonFilter] = useState('');
  const [weekFilter, setWeekFilter] = useState('');
  const [dateRange, setDateRange] = useState({ start: '', end: '' });
  const [fgResultFilter, setFgResultFilter] = useState('');
  const [yardRange, setYardRange] = useState({ min: '', max: '' });
  const [probRange, setProbRange] = useState({ min: '', max: '' });
  const [comparePlayer, setComparePlayer] = useState('');
  const [selectedMetric, setSelectedMetric] = useState("FG_PCT");


  const [playerAttemptsData, setPlayerAttemptsData] = useState([]);
  const [selectedPlayer, setSelectedPlayer] = useState(null);

// FGOE Prediction Form State & Submission

  const [form, setForm] = useState({
    attempt_yards: 42,
    attempt_number_in_game: 2,
    attempt_number_in_season: 18,
    prior_makes_in_game: 1,
    makes_in_a_row: 3,
    misses_in_a_row: 0
  });

  const handlePredict = async () => {
    const res = await fetch("http://localhost:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form)
    });
    const data = await res.json();
    setResponse(data);
  };

// Data Loading

  useEffect(() => {
    if (activeTab === "leaderboard" && leaderboardData.length === 0) {
      Papa.parse("/fg_leaderboard.csv", {
        header: true,
        download: true,
        complete: (results) => {
          const numericFieldsToRound = [
            "expected_makes", "fgoe", "fg_pct", "exp_pct", "fgoe_per_kick"
          ];
          const parsed = results.data.map(row => {
            const cleanRow = {};
            for (const key in row) {
              const val = row[key];
              const parsedVal = parseFloat(val);
              if (numericFieldsToRound.includes(key) && !isNaN(parsedVal)) {
                cleanRow[key] = parsedVal.toFixed(2);
              } else {
                cleanRow[key] = val;
              }
            }
            return cleanRow;
          });
          setLeaderboardData(parsed);
          setFilteredData(parsed);
        }
      });
    }

    if (activeTab === "analysis" && playerAttemptsData.length === 0) {
      Papa.parse("/fgoe_per_attempt.csv", {
        header: true,
        download: true,
        complete: (results) => {
          setPlayerAttemptsData(results.data);
        }
      });
    }
  }, [activeTab, leaderboardData, playerAttemptsData]);

// Leaderboard Search & Sorting

  useEffect(() => {
    if (searchTerm === '') {
      setFilteredData(leaderboardData);
    } else {
      const filtered = leaderboardData.filter(player =>
        player["player_name"]?.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredData(filtered);
      setCurrentPage(1);
    }
  }, [searchTerm, leaderboardData]);

  const handleSort = (key) => {
    let direction = 'ascending';
    if (sortConfig.key === key && sortConfig.direction === 'ascending') {
      direction = 'descending';
    }
    setSortConfig({ key, direction });
  };

  const sortedData = React.useMemo(() => {
    if (!sortConfig.key) return filteredData;
    return [...filteredData].sort((a, b) => {
      const aVal = parseFloat(a[sortConfig.key]);
      const bVal = parseFloat(b[sortConfig.key]);
      if (isNaN(aVal) || isNaN(bVal)) {
        return String(a[sortConfig.key] || '').localeCompare(String(b[sortConfig.key] || ''));
      }
      return sortConfig.direction === 'ascending' ? aVal - bVal : bVal - aVal;
    });
  }, [filteredData, sortConfig]);

  const paginatedData = sortedData.slice(
    (currentPage - 1) * playersPerPage,
    currentPage * playersPerPage
  );

  const totalPages = Math.ceil(filteredData.length / playersPerPage);

// Player Analysis Logic

const renderAnalysisTab = () => {
  const metricOptions = [
    { label: "Makes", key: "Makes" },
    { label: "FG%", key: "FG_PCT" },
    { label: "Avg Pred Prob", key: "AVG_PRED_PROB" },
    { label: "Avg Score", key: "AVG_SCORE" }
  ];

  const playerOptions = [...new Set(playerAttemptsData.map(d => d.player_name))].sort();

  const filtered = playerAttemptsData.filter(row => {
    const seasonMatch = !seasonFilter || row.season === seasonFilter;
    const weekMatch = !weekFilter || row.week === weekFilter;
    const resultMatch = !fgResultFilter || row.field_goal_result === fgResultFilter;
    const dateMatch = (!dateRange.start || new Date(row.game_date) >= new Date(dateRange.start)) &&
                      (!dateRange.end || new Date(row.game_date) <= new Date(dateRange.end));
    const yardMatch = (!yardRange.min || Number(row.attempt_yards) >= Number(yardRange.min)) &&
                      (!yardRange.max || Number(row.attempt_yards) <= Number(yardRange.max));
    const probMatch = (!probRange.min || Number(row.pred_prob) >= Number(probRange.min)) &&
                      (!probRange.max || Number(row.pred_prob) <= Number(probRange.max));
    const playerMatch = row.player_name?.toLowerCase() === selectedPlayer?.toLowerCase() ||
                        row.player_name?.toLowerCase() === comparePlayer?.toLowerCase();
    return seasonMatch && weekMatch && dateMatch && resultMatch && yardMatch && probMatch && playerMatch;
  });

  const aggregated = Object.values(
    filtered.reduce((acc, row) => {
      const name = row.player_name;
      const make = Number(row.make);
      const prob = Number(row.pred_prob);
      const score = Number(row.score);

      if (!acc[name]) {
        acc[name] = { player_name: name, Attempts: 0, Makes: 0, FG_PCT: 0, AVG_PRED_PROB: 0, AVG_SCORE: 0 };
      }

      acc[name].Attempts += 1;
      acc[name].Makes += make;
      acc[name].AVG_PRED_PROB += prob;
      acc[name].AVG_SCORE += score;

      return acc;
    }, {})
  ).map(row => ({
    ...row,
    FG_PCT: row.Makes / row.Attempts || 0,
    AVG_PRED_PROB: row.AVG_PRED_PROB / row.Attempts || 0,
    AVG_SCORE: row.AVG_SCORE / row.Attempts || 0
  }));

  const timeSeriesData = filtered.reduce((acc, row) => {
    const date = row.game_date;
    const name = row.player_name;
    if (!acc[date]) acc[date] = {};
    if (!acc[date][name]) acc[date][name] = { Attempts: 0, Makes: 0, Prob: 0, Score: 0 };
    acc[date][name].Attempts += 1;
    acc[date][name].Makes += Number(row.make);
    acc[date][name].Prob += Number(row.pred_prob);
    acc[date][name].Score += Number(row.score);
    return acc;
  }, {});

  const metricKey = metricOptions.find(opt => opt.key === selectedMetric)?.key || 'FG_PCT';

  const calculateRollingAverage = (arr, key, windowSize = 3) => {
  const result = [];
  for (let i = 0; i < arr.length; i++) {
    let sum = 0;
    let count = 0;
    for (let j = Math.max(0, i - windowSize + 1); j <= i; j++) {
      if (arr[j][key] !== undefined) {
        sum += arr[j][key];
        count++;
      }
    }
    result.push(count > 0 ? sum / count : null);
  }
  return result;
};

const lineChartData = Object.entries(timeSeriesData).map(([date, players]) => {
  const entry = { game_date: date };
  [selectedPlayer, comparePlayer].forEach(player => {
    if (player && players[player]) {
      const { Attempts, Makes, Prob, Score } = players[player];
      entry[`${player}_Makes`] = Makes;
      entry[`${player}_FG_PCT`] = Makes / Attempts;
      entry[`${player}_AVG_PRED_PROB`] = Prob / Attempts;
      entry[`${player}_AVG_SCORE`] = Score / Attempts;
    }
  });
  return entry;
}).sort((a, b) => new Date(a.game_date) - new Date(b.game_date));

const rollingLineChartData = lineChartData.map((d, i, fullArr) => {
  const updated = { ...d };
  [selectedPlayer, comparePlayer].forEach(player => {
    if (player) {
      ["Makes", "FG_PCT", "AVG_PRED_PROB", "AVG_SCORE"].forEach(metric => {
        const key = `${player}_${metric}`;
        const series = fullArr.map(row => row[key]);
        const rolled = calculateRollingAverage(fullArr, key);
        updated[key] = rolled[i];
      });
    }
  });
  return updated;
});

// App Layout and Visuals

  return (
    <div>
      <h1>Player Analysis</h1>

      {/* Player selectors and filters */}
      <div style={{ marginBottom: '1rem' }}>
        <select value={selectedPlayer || ''} onChange={e => setSelectedPlayer(e.target.value)} style={{ marginRight: '1rem' }}>
          <option value="">Select Player A</option>
          {playerOptions.map(p => <option key={p} value={p}>{p}</option>)}
        </select>
        <select value={comparePlayer} onChange={e => setComparePlayer(e.target.value)}>
          <option value="">Compare with Player B (optional)</option>
          {playerOptions.map(p => <option key={p} value={p}>{p}</option>)}
        </select>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginBottom: '1rem' }}>
        <select value={seasonFilter} onChange={e => setSeasonFilter(e.target.value)}>
          <option value="">All Seasons</option>
          {[...new Set(playerAttemptsData.map(d => d.season))].sort().map(s => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <select value={weekFilter} onChange={e => setWeekFilter(e.target.value)}>
          <option value="">All Weeks</option>
          {[...new Set(playerAttemptsData.map(d => d.week))].sort().map(w => (
            <option key={w} value={w}>{w}</option>
          ))}
        </select>
        <input type="date" placeholder="Start Date" value={dateRange.start} onChange={e => setDateRange({ ...dateRange, start: e.target.value })} />
        <input type="date" placeholder="End Date" value={dateRange.end} onChange={e => setDateRange({ ...dateRange, end: e.target.value })} />
        <select value={fgResultFilter} onChange={e => setFgResultFilter(e.target.value)}>
          <option value="">All Results</option>
          {[...new Set(playerAttemptsData.map(d => d.field_goal_result))].map(r => (
            <option key={r} value={r}>{r}</option>
          ))}
        </select>
        <input type="number" placeholder="Min Yards" value={yardRange.min} onChange={e => setYardRange({ ...yardRange, min: e.target.value })} />
        <input type="number" placeholder="Max Yards" value={yardRange.max} onChange={e => setYardRange({ ...yardRange, max: e.target.value })} />
        <input type="number" placeholder="Min Prob" value={probRange.min} step="0.01" onChange={e => setProbRange({ ...probRange, min: e.target.value })} />
        <input type="number" placeholder="Max Prob" value={probRange.max} step="0.01" onChange={e => setProbRange({ ...probRange, max: e.target.value })} />
      </div>

      {/* Metric selection for visuals */}
      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem' }}>
        <label style={{ marginRight: '0.5rem' }}>Select Metric:</label>
        <select value={selectedMetric} onChange={e => setSelectedMetric(e.target.value)}>
          {metricOptions.map(opt => (
            <option key={opt.key} value={opt.key}>{opt.label}</option>
          ))}
        </select>
      </div>

      {/* Aggregated Table */}
      <table style={{ borderCollapse: 'collapse', width: '100%', marginBottom: '2rem' }}>
        <thead>
          <tr style={{ backgroundColor: '#002244', color: 'white' }}>
            <th style={{ padding: '8px', border: '1px solid #ddd' }}>Player</th>
            <th style={{ padding: '8px', border: '1px solid #ddd' }}>Attempts</th>
            <th style={{ padding: '8px', border: '1px solid #ddd' }}>Makes</th>
            <th style={{ padding: '8px', border: '1px solid #ddd' }}>FG%</th>
            <th style={{ padding: '8px', border: '1px solid #ddd' }}>Avg Pred Prob</th>
            <th style={{ padding: '8px', border: '1px solid #ddd' }}>Avg Score</th>
          </tr>
        </thead>
        <tbody>
          {aggregated.map((row, i) => (
            <tr key={i}>
              <td style={{ border: '1px solid #ddd', textAlign: 'center' }}>{row.player_name}</td>
              <td style={{ border: '1px solid #ddd', textAlign: 'center' }}>{row.Attempts}</td>
              <td style={{ border: '1px solid #ddd', textAlign: 'center' }}>{row.Makes}</td>
              <td style={{ border: '1px solid #ddd', textAlign: 'center' }}>{row.FG_PCT.toFixed(2)}</td>
              <td style={{ border: '1px solid #ddd', textAlign: 'center' }}>{row.AVG_PRED_PROB.toFixed(2)}</td>
              <td style={{ border: '1px solid #ddd', textAlign: 'center' }}>{row.AVG_SCORE.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Bar Chart */}
      <h3 style={{ marginTop: '2rem' }}>Metric Comparison (Bar)</h3>
      <ResponsiveContainer width="100%" height={300}>
      <BarChart data={aggregated}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="player_name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar
          dataKey={selectedMetric}
          fill="#002244"
          shape={(props) => {
            const { x, y, width, height, payload } = props;
            const color =
              payload.player_name === comparePlayer ? "#FF6600" : "#002244";
            return <rect x={x} y={y} width={width} height={height} fill={color} />;
          }}
        />
      </BarChart>
      </ResponsiveContainer>

      {/* Line Chart */}
      <h3 style={{ marginTop: '2rem' }}>Metric Over Time (Line)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={rollingLineChartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="game_date" />
          <YAxis />
          <Tooltip />
          <Legend />
          {selectedPlayer && (
            <Line type="monotone" dataKey={`${selectedPlayer}_${selectedMetric}`} stroke="#002244" strokeWidth={2} />
          )}
          {comparePlayer && (
            <Line type="monotone" dataKey={`${comparePlayer}_${selectedMetric}`} stroke="#FF6600" strokeWidth={2} />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

  if (!showApp) {
    return (
      <div
        style={{
          height: '100vh',
          backgroundImage: `url(${broncosLogo})`,
          backgroundSize: 'contain',
          backgroundRepeat: 'no-repeat',
          backgroundPosition: 'center',
          backgroundColor: '#001f3f',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column',
          cursor: 'pointer'
        }}
        onClick={() => setShowApp(true)}
      >
        <p style={{ color: 'white' }}>(Click anywhere to enter)</p>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'Arial' }}>
      {/* Sidebar */}
      <div style={{
        width: '220px',
        backgroundColor: '#002244',
        color: 'white',
        padding: '1rem',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center'
      }}>
        <h1 style={{ color: '#FF6600', margin: '0' }}>The Stable</h1>
        <img src={broncosPrimaryLogo} alt="Broncos Logo" style={{ width: '100px', margin: '0' }} />
        {['predictor', 'leaderboard', 'analysis'].map(tab => (
          <div
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              margin: '1rem 0',
              cursor: 'pointer',
              fontWeight: activeTab === tab ? 'bold' : 'normal'
            }}
          >
            {tab === 'predictor' ? 'FGOE Predictor' : tab === 'leaderboard' ? 'Leaderboard' : 'Player Analysis'}
          </div>
        ))}
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, padding: '2rem', overflowY: 'scroll' }}>
        {activeTab === 'predictor' && (
          <>
            <h1>Field Goal Over Expected (FGOE) Predictor</h1>
            <div style={{ marginBottom: "1rem" }}>
              {Object.entries(form).map(([key, value]) => (
                <div key={key} style={{ marginBottom: '10px' }}>
                  <label style={{ marginRight: '10px' }}>{key.replace(/_/g, ' ')}:</label>
                  <input
                    type="number"
                    value={value}
                    onChange={(e) =>
                      setForm({ ...form, [key]: parseInt(e.target.value) || 0 })
                    }
                  />
                </div>
              ))}
            </div>
            <button onClick={handlePredict} style={{ padding: "0.5rem 1rem" }}>
              Predict FGOE
            </button>
            {response && (
              <div style={{ marginTop: "1.5rem" }}>
                <h3>FGOE Probability:</h3>
                <p style={{ fontSize: "1.5rem", fontWeight: "bold", color: "black" }}>
                  {response.fgoe_probability}
                </p>
              </div>
            )}
          </>
        )}

        {activeTab === 'leaderboard' && (
          <div>
            <h1>Leaderboard</h1>
            <input
              type="text"
              placeholder="Search player name"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                marginBottom: '1rem',
                padding: '0.5rem',
                width: '100%',
                border: '1px solid #ccc',
                borderRadius: '4px'
              }}
            />
            {filteredData.length > 0 ? (
              <>
                <table style={{ borderCollapse: 'collapse', width: '100%' }}>
                  <thead>
                    <tr>
                      {Object.keys(filteredData[0]).map((key) => (
                        <th
                          key={key}
                          onClick={() => handleSort(key)}
                          style={{
                            border: '1px solid #ddd',
                            padding: '8px',
                            backgroundColor: '#002244',
                            color: 'white',
                            cursor: 'pointer'
                          }}
                        >
                          {key.replace(/_/g, ' ')} {sortConfig.key === key ? (sortConfig.direction === 'ascending' ? '↑' : '↓') : ''}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedData.map((row, idx) => (
                      <tr key={idx}>
                        {Object.values(row).map((value, i) => (
                          <td
                            key={i}
                            style={{
                              border: '1px solid #ddd',
                              padding: '8px',
                              textAlign: 'center'
                            }}
                          >
                            {value}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ marginTop: '1rem', textAlign: 'center' }}>
                  {Array.from({ length: totalPages }, (_, i) => (
                    <button
                      key={i + 1}
                      onClick={() => setCurrentPage(i + 1)}
                      style={{
                        margin: '0 5px',
                        padding: '5px 10px',
                        backgroundColor: i + 1 === currentPage ? '#FF6600' : '#ddd',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: 'pointer'
                      }}
                    >
                      {i + 1}
                    </button>
                  ))}
                </div>
              </>
            ) : (
              <p>Loading leaderboard...</p>
            )}
          </div>
        )}

        {activeTab === 'analysis' && renderAnalysisTab()}
      </div>
    </div>
  );
}

export default App;