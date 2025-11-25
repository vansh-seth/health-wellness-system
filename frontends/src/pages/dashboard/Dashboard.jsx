import { useState, useEffect } from 'react';

const API_URL = 'http://localhost:8000';

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [anomalies, setAnomalies] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const token = localStorage.getItem('token');
      const headers = { Authorization: `Bearer ${token}` };

      // Fetch all data in parallel
      const [statsRes, predictionsRes, recommendationsRes, anomaliesRes] = await Promise.allSettled([
        fetch(`${API_URL}/dashboard/stats?days=7`, { headers }).then(r => r.json()),
        fetch(`${API_URL}/ai/predictions`, { headers }).then(r => r.json()),
        fetch(`${API_URL}/ai/recommendations`, { headers }).then(r => r.json()),
        fetch(`${API_URL}/ai/anomalies`, { headers }).then(r => r.json())
      ]);

      if (statsRes.status === 'fulfilled') setStats(statsRes.value);
      if (predictionsRes.status === 'fulfilled') setPredictions(predictionsRes.value);
      if (recommendationsRes.status === 'fulfilled') setRecommendations(recommendationsRes.value);
      if (anomaliesRes.status === 'fulfilled') setAnomalies(anomaliesRes.value);

      setLoading(false);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-purple-600 mx-auto mb-4"></div>
          <p className="text-gray-600 text-lg">Loading your health insights...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">Your Health Dashboard</h1>
          <p className="text-gray-600">AI-powered insights and personalized recommendations</p>
        </div>

        {anomalies && anomalies.is_anomaly && (
          <div className={`mb-6 p-4 rounded-lg border-l-4 ${
            anomalies.severity === 'critical' ? 'bg-red-100 border-red-400' :
            anomalies.severity === 'high' ? 'bg-orange-100 border-orange-400' :
            'bg-yellow-100 border-yellow-400'
          }`}>
            <div className="flex items-center">
              <span className="text-2xl mr-3">⚠️</span>
              <div>
                <h3 className="font-semibold text-gray-800">
                  Health Alert: {anomalies.severity.toUpperCase()} Anomaly Detected
                </h3>
                {anomalies.insights && anomalies.insights.length > 0 && (
                  <ul className="mt-2 space-y-1">
                    {anomalies.insights.map((insight, idx) => (
                      <li key={idx} className="text-sm text-gray-700">{insight}</li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>
        )}

        <div className="mb-6 border-b border-gray-300">
          <div className="flex space-x-8">
            {['overview', 'predictions', 'recommendations'].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab
                    ? 'border-purple-600 text-purple-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {activeTab === 'overview' && stats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-blue-500">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-gray-600 font-semibold">Sleep</h3>
                <span className="text-3xl">😴</span>
              </div>
              <div className="space-y-2">
                <p className="text-3xl font-bold text-gray-800">
                  {stats.avg_sleep_hours?.toFixed(1) || '0.0'}h
                </p>
                <p className="text-sm text-gray-500">Avg Duration</p>
                {stats.avg_sleep_quality && (
                  <p className="text-sm text-gray-600">Quality: {stats.avg_sleep_quality.toFixed(1)}/10</p>
                )}
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-green-500">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-gray-600 font-semibold">Mood</h3>
                <span className="text-3xl">😊</span>
              </div>
              <div className="space-y-2">
                <p className="text-3xl font-bold text-gray-800">
                  {stats.avg_mood_score?.toFixed(1) || '0.0'}/10
                </p>
                <p className="text-sm text-gray-500">Avg Score</p>
                {stats.avg_stress_level && (
                  <p className="text-sm text-gray-600">Stress: {stats.avg_stress_level.toFixed(1)}/10</p>
                )}
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-orange-500">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-gray-600 font-semibold">Nutrition</h3>
                <span className="text-3xl">🍽️</span>
              </div>
              <div className="space-y-2">
                <p className="text-3xl font-bold text-gray-800">
                  {stats.total_calories_consumed?.toLocaleString() || '0'}
                </p>
                <p className="text-sm text-gray-500">Total Calories</p>
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-6 border-l-4 border-purple-500">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-gray-600 font-semibold">Exercise</h3>
                <span className="text-3xl">🏃</span>
              </div>
              <div className="space-y-2">
                <p className="text-3xl font-bold text-gray-800">
                  {stats.total_calories_burned?.toLocaleString() || '0'}
                </p>
                <p className="text-sm text-gray-500">Calories Burned</p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'predictions' && (
          <div className="bg-white rounded-xl shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">🔮 Tomorrow's Predictions</h2>
            
            {!predictions ? (
              <div className="text-center py-12">
                <div className="text-6xl mb-4">📊</div>
                <h3 className="text-xl font-semibold text-gray-700 mb-2">Predictions Unavailable</h3>
                <p className="text-gray-600 mb-4">
                  We need at least 7 days of sleep and mood data to generate predictions.
                </p>
                <p className="text-sm text-gray-500">
                  Keep logging your health data daily to unlock AI-powered predictions!
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="border-2 border-blue-200 rounded-lg p-6 bg-blue-50">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">Sleep Quality</h3>
                  <p className="text-4xl font-bold text-blue-600 mb-2">
                    {predictions.predictions.sleep_quality.value.toFixed(1)}/10
                  </p>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-blue-600 h-2 rounded-full" 
                      style={{ width: `${predictions.predictions.sleep_quality.value * 10}%` }}></div>
                  </div>
                </div>

                <div className="border-2 border-green-200 rounded-lg p-6 bg-green-50">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">Mood Score</h3>
                  <p className="text-4xl font-bold text-green-600 mb-2">
                    {predictions.predictions.mood_score.value.toFixed(1)}/10
                  </p>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-green-600 h-2 rounded-full"
                      style={{ width: `${predictions.predictions.mood_score.value * 10}%` }}></div>
                  </div>
                </div>

                <div className="border-2 border-orange-200 rounded-lg p-6 bg-orange-50">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">Stress Level</h3>
                  <p className="text-4xl font-bold text-orange-600 mb-2">
                    {predictions.predictions.stress_level.value.toFixed(1)}/10
                  </p>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-orange-600 h-2 rounded-full"
                      style={{ width: `${predictions.predictions.stress_level.value * 10}%` }}></div>
                  </div>
                </div>

                <div className="border-2 border-purple-200 rounded-lg p-6 bg-purple-50">
                  <h3 className="text-lg font-semibold text-gray-800 mb-4">Exercise Likelihood</h3>
                  <p className="text-4xl font-bold text-purple-600 mb-2">
                    {(predictions.predictions.exercise_probability.value * 100).toFixed(0)}%
                  </p>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div className="bg-purple-600 h-2 rounded-full"
                      style={{ width: `${predictions.predictions.exercise_probability.value * 100}%` }}></div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'recommendations' && recommendations && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">🍽️ Meal Recommendations</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {recommendations.meals.map((rec, idx) => (
                  <div key={idx} className="border-2 border-orange-200 rounded-lg p-6">
                    <h3 className="text-lg font-semibold mb-3">{rec.meal.name}</h3>
                    <div className="space-y-2 mb-4 text-sm">
                      <div className="flex justify-between">
                        <span>Calories:</span>
                        <span className="font-semibold">{rec.meal.calories}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Protein:</span>
                        <span className="font-semibold">{rec.meal.protein}g</span>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 border-t pt-3">💡 {rec.reasoning}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">🏃 Exercise Recommendations</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {recommendations.exercises.map((rec, idx) => (
                  <div key={idx} className="border-2 border-purple-200 rounded-lg p-6">
                    <h3 className="text-lg font-semibold mb-3">{rec.exercise.name}</h3>
                    <div className="space-y-2 mb-4 text-sm">
                      <div className="flex justify-between">
                        <span>Duration:</span>
                        <span className="font-semibold">{rec.exercise.duration} min</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Type:</span>
                        <span className="font-semibold">{rec.exercise.type}</span>
                      </div>
                    </div>
                    <p className="text-sm text-gray-600 border-t pt-3">💡 {rec.reasoning}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-xl shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6">💡 Wellness Tips</h2>
              <div className="space-y-4">
                {recommendations.wellness_tips.map((rec, idx) => (
                  <div key={idx} className={`border-l-4 rounded-lg p-6 ${
                    rec.priority === 'high' ? 'border-red-500 bg-red-50' : 'border-green-500 bg-green-50'
                  }`}>
                    <h3 className="font-semibold text-gray-800 mb-2">{rec.tip.tip}</h3>
                    <p className="text-sm text-gray-600">💡 {rec.reasoning}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
