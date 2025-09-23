// src/pages/dashboard/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { api } from '../../services/api';

export const Dashboard = () => {
  const [stats, setStats] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const statsData = await api.getDashboardStats();
      setStats(statsData);
    } catch (error) {
      console.error('Error fetching stats:', error);
      setError('Failed to load dashboard stats');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="main-content">
        <div className="loading-container">Loading dashboard...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="main-content">
        <div className="error-container">
          <p>{error}</p>
          <button onClick={fetchStats}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="main-content">
      <div className="dashboard">
        <h2>Dashboard</h2>
        <div className="stats-grid">
          <div className="stat-card">
            <h3>Total Logs</h3>
            <p className="stat-number">{stats.total_logs || 0}</p>
          </div>
          <div className="stat-card">
            <h3>Meals Logged</h3>
            <p className="stat-number">{stats.meals_logged || 0}</p>
          </div>
          <div className="stat-card">
            <h3>Exercises</h3>
            <p className="stat-number">{stats.exercises_logged || 0}</p>
          </div>
          <div className="stat-card">
            <h3>Sleep Logs</h3>
            <p className="stat-number">{stats.sleep_logs || 0}</p>
          </div>
          <div className="stat-card">
            <h3>Avg Sleep Hours</h3>
            <p className="stat-number">{(stats.avg_sleep_hours || 0).toFixed(1)}</p>
          </div>
          <div className="stat-card">
            <h3>Avg Mood Score</h3>
            <p className="stat-number">{(stats.avg_mood_score || 0).toFixed(1)}/10</p>
          </div>
          <div className="stat-card">
            <h3>Calories Consumed</h3>
            <p className="stat-number">{stats.total_calories_consumed || 0}</p>
          </div>
          <div className="stat-card">
            <h3>Calories Burned</h3>
            <p className="stat-number">{stats.total_calories_burned || 0}</p>
          </div>
        </div>

        {stats.total_logs === 0 && (
          <div className="welcome-message">
            <h3>Welcome to your Health Dashboard!</h3>
            <p>Start logging your health data to see insights and trends.</p>
            <div className="quick-actions">
              <Link to="/log" className="action-button">Log Your First Entry</Link>
            </div>
          </div>
        )}

        {stats.total_logs > 0 && (
          <div className="dashboard-insights">
            <h3>Quick Insights</h3>
            <div className="insights-grid">
              {stats.avg_sleep_hours < 7 && (
                <div className="insight-card warning">
                  <h4>💤 Sleep Recommendation</h4>
                  <p>Your average sleep is {(stats.avg_sleep_hours || 0).toFixed(1)} hours. Consider aiming for 7-9 hours for better health.</p>
                </div>
              )}
              {stats.exercises_logged === 0 && (
                <div className="insight-card info">
                  <h4>🏃‍♂️ Stay Active</h4>
                  <p>No exercises logged yet. Try to get at least 30 minutes of physical activity daily.</p>
                </div>
              )}
              {stats.avg_mood_score > 0 && stats.avg_mood_score < 6 && (
                <div className="insight-card warning">
                  <h4>😊 Mood Check</h4>
                  <p>Your average mood score is {(stats.avg_mood_score || 0).toFixed(1)}/10. Consider stress management techniques.</p>
                </div>
              )}
              {stats.total_logs > 10 && (
                <div className="insight-card success">
                  <h4>🎉 Great Progress!</h4>
                  <p>You've logged {stats.total_logs} entries. Keep up the consistent health tracking!</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};