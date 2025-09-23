// pages/logging/MoodForm.jsx
import React, { useState } from 'react';
import { api } from '../../services/api';

export const MoodForm = () => {
  const [formData, setFormData] = useState({
    mood_score: '7',
    stress_level: '5',
    energy_level: '7',
    notes: ''
  });
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setSuccess(false);
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const moodData = {
        mood_score: parseInt(formData.mood_score),
        stress_level: parseInt(formData.stress_level),
        energy_level: parseInt(formData.energy_level),
        notes: formData.notes.trim()
      };

      await api.logMood(moodData);
      setSuccess(true);
      setFormData({
        mood_score: '7',
        stress_level: '5',
        energy_level: '7',
        notes: ''
      });
    } catch (error) {
      console.error('Error logging mood:', error);
      setError(error.message || 'Failed to log mood');
    } finally {
      setLoading(false);
    }
  };

  const getMoodEmoji = (score) => {
    if (score >= 9) return '😄';
    if (score >= 7) return '😊';
    if (score >= 5) return '😐';
    if (score >= 3) return '😔';
    return '😢';
  };

  const getStressColor = (level) => {
    if (level >= 8) return '#ff4757';
    if (level >= 6) return '#ffa502';
    if (level >= 4) return '#ffb833';
    return '#26de81';
  };

  const getEnergyColor = (level) => {
    if (level >= 8) return '#26de81';
    if (level >= 6) return '#ffb833';
    if (level >= 4) return '#ffa502';
    return '#ff4757';
  };

  return (
    <div className="form-wrapper">
      <h3>Log Mood</h3>
      {success && <div className="success-message">Mood logged successfully!</div>}
      {error && <div className="error-message">{error}</div>}
      
      <form onSubmit={handleSubmit} className="log-form">
        <label>Mood Score (1-10):</label>
        <div className="range-input">
          <input
            type="range"
            name="mood_score"
            min="1"
            max="10"
            value={formData.mood_score}
            onChange={handleChange}
          />
          <span className="range-value">
            {getMoodEmoji(parseInt(formData.mood_score))} {formData.mood_score}/10
          </span>
        </div>

        <label>Stress Level (1-10):</label>
        <div className="range-input">
          <input
            type="range"
            name="stress_level"
            min="1"
            max="10"
            value={formData.stress_level}
            onChange={handleChange}
          />
          <span 
            className="range-value"
            style={{ color: getStressColor(parseInt(formData.stress_level)) }}
          >
            {formData.stress_level}/10
          </span>
        </div>

        <label>Energy Level (1-10):</label>
        <div className="range-input">
          <input
            type="range"
            name="energy_level"
            min="1"
            max="10"
            value={formData.energy_level}
            onChange={handleChange}
          />
          <span 
            className="range-value"
            style={{ color: getEnergyColor(parseInt(formData.energy_level)) }}
          >
            {formData.energy_level}/10
          </span>
        </div>

        <label>Additional Notes (optional):</label>
        <textarea
          name="notes"
          placeholder="How are you feeling today? Any specific thoughts or events?"
          value={formData.notes}
          onChange={handleChange}
          rows="4"
          maxLength="500"
        />
        <small className="char-count">{formData.notes.length}/500 characters</small>
        
        <button type="submit" disabled={loading}>
          {loading ? 'Logging Mood...' : 'Log Mood'}
        </button>
      </form>
    </div>
  );
};