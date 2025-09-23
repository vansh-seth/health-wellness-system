// pages/history/History.jsx
import React, { useState, useEffect } from 'react';
import { api } from '../../services/api';

export const History = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState('all');
  const [days, setDays] = useState(7);

  useEffect(() => {
    fetchLogs();
  }, [filter, days]);

  const fetchLogs = async () => {
    try {
      setLoading(true);
      const logType = filter === 'all' ? null : filter;
      const logsData = await api.getHealthLogs(logType, days);
      setLogs(logsData);
    } catch (error) {
      console.error('Error fetching logs:', error);
      setError('Failed to load health history');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (timestamp) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      weekday: 'short',
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const renderLogContent = (log) => {
    switch (log.log_type) {
      case 'meal':
        return (
          <div className="log-details">
            <h4>{log.data.meal_name}</h4>
            <p><strong>Type:</strong> {log.data.meal_type}</p>
            <p><strong>Calories:</strong> {log.data.calories}</p>
            {(log.data.protein > 0 || log.data.carbs > 0 || log.data.fat > 0) && (
              <p><strong>Macros:</strong> {log.data.protein}g protein, {log.data.carbs}g carbs, {log.data.fat}g fat</p>
            )}
          </div>
        );
      
      case 'exercise':
        return (
          <div className="log-details">
            <h4>{log.data.exercise_name}</h4>
            <p><strong>Type:</strong> {log.data.exercise_type}</p>
            <p><strong>Duration:</strong> {log.data.duration_minutes} minutes</p>
            {log.data.calories_burned > 0 && (
              <p><strong>Calories Burned:</strong> {log.data.calories_burned}</p>
            )}
          </div>
        );
      
      case 'sleep':
        return (
          <div className="log-details">
            <h4>Sleep Session</h4>
            <p><strong>Duration:</strong> {log.data.sleep_duration_hours?.toFixed(1)} hours</p>
            <p><strong>Quality:</strong> {log.data.sleep_quality}/10</p>
            <p><strong>Bedtime:</strong> {formatTime(log.data.bedtime)}</p>
            <p><strong>Wake Time:</strong> {formatTime(log.data.wake_time)}</p>
          </div>
        );
      
      case 'mood':
        const getMoodEmoji = (score) => {
          if (score >= 9) return '😄';
          if (score >= 7) return '😊';
          if (score >= 5) return '😐';
          if (score >= 3) return '😔';
          return '😢';
        };

        return (
          <div className="log-details">
            <h4>Mood Check-in {getMoodEmoji(log.data.mood_score)}</h4>
            <p><strong>Mood:</strong> {log.data.mood_score}/10</p>
            <p><strong>Stress:</strong> {log.data.stress_level}/10</p>
            <p><strong>Energy:</strong> {log.data.energy_level}/10</p>
            {log.data.notes && <p><strong>Notes:</strong> {log.data.notes}</p>}
          </div>
        );
      
      default:
        return <div>Unknown log type</div>;
    }
  };

  if (loading) {
    return (
      <div className="main-content">
        <div className="loading-container">Loading health history...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="main-content">
        <div className="error-container">
          <p>{error}</p>
          <button onClick={fetchLogs}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="main-content">
      <div className="history-container">
        <h2>Health History</h2>
        
        <div className="history-filters">
          <div className="filter-group">
            <label>Filter by Type:</label>
            <select value={filter} onChange={(e) => setFilter(e.target.value)}>
              <option value="all">All Logs</option>
              <option value="meal">Meals</option>
              <option value="exercise">Exercises</option>
              <option value="sleep">Sleep</option>
              <option value="mood">Mood</option>
            </select>
          </div>
          
          <div className="filter-group">
            <label>Time Period:</label>
            <select value={days} onChange={(e) => setDays(parseInt(e.target.value))}>
              <option value={7}>Last 7 days</option>
              <option value={14}>Last 2 weeks</option>
              <option value={30}>Last 30 days</option>
              <option value={90}>Last 3 months</option>
            </select>
          </div>
        </div>

        {logs.length === 0 ? (
          <div className="no-logs">
            <h3>No logs found</h3>
            <p>
              {filter === 'all' 
                ? `No health data logged in the last ${days} days.`
                : `No ${filter} logs found in the last ${days} days.`
              }
            </p>
            <a href="/log" className="action-button">Log Some Data</a>
          </div>
        ) : (
          <div className="logs-list">
            <p className="logs-count">
              Showing {logs.length} log{logs.length !== 1 ? 's' : ''} from the last {days} days
            </p>
            
            {logs.map((log, index) => (
              <div key={log._id || index} className={`log-item ${log.log_type}`}>
                <div className="log-header">
                  <span className={`log-type-badge ${log.log_type}`}>
                    {log.log_type.toUpperCase()}
                  </span>
                  <div className="log-timestamp">
                    <span className="log-date">{formatDate(log.timestamp)}</span>
                    <span className="log-time">{formatTime(log.timestamp)}</span>
                  </div>
                </div>
                
                <div className="log-content">
                  {renderLogContent(log)}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};