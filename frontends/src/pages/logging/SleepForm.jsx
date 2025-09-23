// pages/logging/SleepForm.jsx
import React, { useState } from 'react';
import { api } from '../../services/api';

export const SleepForm = () => {
  const [formData, setFormData] = useState({
    bedtime: '',
    wake_time: '',
    sleep_quality: '7'
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
      // Validate that wake_time is after bedtime
      const bedtime = new Date(formData.bedtime);
      const wakeTime = new Date(formData.wake_time);
      
      if (wakeTime <= bedtime) {
        setError('Wake time must be after bedtime');
        setLoading(false);
        return;
      }

      const sleepData = {
        bedtime: bedtime.toISOString(),
        wake_time: wakeTime.toISOString(),
        sleep_quality: parseInt(formData.sleep_quality)
      };

      await api.logSleep(sleepData);
      setSuccess(true);
      setFormData({
        bedtime: '',
        wake_time: '',
        sleep_quality: '7'
      });
    } catch (error) {
      console.error('Error logging sleep:', error);
      setError(error.message || 'Failed to log sleep');
    } finally {
      setLoading(false);
    }
  };

  // Calculate sleep duration for preview
  const calculateSleepDuration = () => {
    if (formData.bedtime && formData.wake_time) {
      const bedtime = new Date(formData.bedtime);
      const wakeTime = new Date(formData.wake_time);
      
      if (wakeTime > bedtime) {
        const durationMs = wakeTime - bedtime;
        const hours = Math.floor(durationMs / (1000 * 60 * 60));
        const minutes = Math.floor((durationMs % (1000 * 60 * 60)) / (1000 * 60));
        return `${hours}h ${minutes}m`;
      }
    }
    return null;
  };

  return (
    <div className="form-wrapper">
      <h3>Log Sleep</h3>
      {success && <div className="success-message">Sleep logged successfully!</div>}
      {error && <div className="error-message">{error}</div>}
      
      <form onSubmit={handleSubmit} className="log-form">
        <label>Bedtime:</label>
        <input
          type="datetime-local"
          name="bedtime"
          value={formData.bedtime}
          onChange={handleChange}
          required
        />
        
        <label>Wake Time:</label>
        <input
          type="datetime-local"
          name="wake_time"
          value={formData.wake_time}
          onChange={handleChange}
          required
        />
        
        {calculateSleepDuration() && (
          <div className="sleep-duration">
            <strong>Sleep Duration: {calculateSleepDuration()}</strong>
          </div>
        )}
        
        <label>Sleep Quality (1-10):</label>
        <div className="range-input">
          <input
            type="range"
            name="sleep_quality"
            min="1"
            max="10"
            value={formData.sleep_quality}
            onChange={handleChange}
          />
          <span className="range-value">{formData.sleep_quality}/10</span>
        </div>
        
        <button type="submit" disabled={loading}>
          {loading ? 'Logging Sleep...' : 'Log Sleep'}
        </button>
      </form>
    </div>
  );
};