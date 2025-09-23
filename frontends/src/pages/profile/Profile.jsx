// pages/profile/Profile.jsx
import React, { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import { api } from '../../services/api';

const Profile = () => {
  const { user, fetchProfile } = useAuth();
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  
  const [formData, setFormData] = useState({
    first_name: '',
    last_name: '',
    age: '',
    gender: '',
    height: '',
    weight: '',
    activity_level: '',
    health_goals: [],
    medical_conditions: [],
    allergies: []
  });

  useEffect(() => {
    if (user) {
      setFormData({
        first_name: user.first_name || '',
        last_name: user.last_name || '',
        age: user.age || '',
        gender: user.gender || '',
        height: user.height || '',
        weight: user.weight || '',
        activity_level: user.activity_level || '',
        health_goals: user.health_goals || [],
        medical_conditions: user.medical_conditions || [],
        allergies: user.allergies || []
      });
    }
  }, [user]);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError('');
    setSuccess(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const updates = {
        ...formData,
        age: parseInt(formData.age),
        height: parseFloat(formData.height),
        weight: parseFloat(formData.weight)
      };

      await api.updateProfile(updates);
      await fetchProfile();
      setSuccess(true);
      setIsEditing(false);
    } catch (error) {
      console.error('Error updating profile:', error);
      setError(error.message || 'Failed to update profile');
    } finally {
      setLoading(false);
    }
  };

  const calculateBMI = () => {
    if (user && user.height && user.weight) {
      const bmi = user.weight / ((user.height / 100) ** 2);
      return bmi.toFixed(1);
    }
    return 'N/A';
  };

  const getBMICategory = (bmi) => {
    if (bmi === 'N/A') return '';
    const bmiNum = parseFloat(bmi);
    if (bmiNum < 18.5) return 'Underweight';
    if (bmiNum < 25) return 'Normal weight';
    if (bmiNum < 30) return 'Overweight';
    return 'Obese';
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  if (!user) {
    return (
      <div className="main-content">
        <div className="loading-container">Loading profile...</div>
      </div>
    );
  }

  return (
    <div className="main-content">
      <div className="profile-container">
        <div className="profile-header">
          <h2>My Profile</h2>
          {!isEditing && (
            <button 
              className="edit-button" 
              onClick={() => setIsEditing(true)}
            >
              Edit Profile
            </button>
          )}
        </div>

        {success && <div className="success-message">Profile updated successfully!</div>}
        {error && <div className="error-message">{error}</div>}

        {isEditing ? (
          <form onSubmit={handleSubmit} className="profile-edit-form">
            <div className="form-section">
              <h3>Personal Information</h3>
              
              <div className="form-row">
                <input
                  type="text"
                  name="first_name"
                  placeholder="First Name"
                  value={formData.first_name}
                  onChange={handleChange}
                  required
                />
                <input
                  type="text"
                  name="last_name"
                  placeholder="Last Name"
                  value={formData.last_name}
                  onChange={handleChange}
                  required
                />
              </div>

              <div className="form-row">
                <input
                  type="number"
                  name="age"
                  placeholder="Age"
                  value={formData.age}
                  onChange={handleChange}
                  min="1"
                  max="120"
                  required
                />
                <select name="gender" value={formData.gender} onChange={handleChange}>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other</option>
                </select>
              </div>
            </div>

            <div className="form-section">
              <h3>Physical Stats</h3>
              
              <div className="form-row">
                <input
                  type="number"
                  step="0.1"
                  name="height"
                  placeholder="Height (cm)"
                  value={formData.height}
                  onChange={handleChange}
                  min="50"
                  max="250"
                  required
                />
                <input
                  type="number"
                  step="0.1"
                  name="weight"
                  placeholder="Weight (kg)"
                  value={formData.weight}
                  onChange={handleChange}
                  min="20"
                  max="300"
                  required
                />
              </div>

              <label>Activity Level:</label>
              <select name="activity_level" value={formData.activity_level} onChange={handleChange}>
                <option value="sedentary">Sedentary</option>
                <option value="lightly_active">Lightly Active</option>
                <option value="moderately_active">Moderately Active</option>
                <option value="very_active">Very Active</option>
              </select>
            </div>

            <div className="form-actions">
              <button type="submit" disabled={loading}>
                {loading ? 'Saving...' : 'Save Changes'}
              </button>
              <button 
                type="button" 
                className="cancel-button"
                onClick={() => {
                  setIsEditing(false);
                  setError('');
                  setSuccess(false);
                }}
              >
                Cancel
              </button>
            </div>
          </form>
        ) : (
          <div className="profile-info">
            <div className="profile-section">
              <h3>Personal Information</h3>
              <div className="info-grid">
                <div className="info-item">
                  <label>Name:</label>
                  <span>{user.first_name} {user.last_name}</span>
                </div>
                <div className="info-item">
                  <label>Email:</label>
                  <span>{user.email}</span>
                </div>
                <div className="info-item">
                  <label>Age:</label>
                  <span>{user.age} years</span>
                </div>
                <div className="info-item">
                  <label>Gender:</label>
                  <span style={{ textTransform: 'capitalize' }}>{user.gender}</span>
                </div>
              </div>
            </div>
            
            <div className="profile-section">
              <h3>Physical Stats</h3>
              <div className="info-grid">
                <div className="info-item">
                  <label>Height:</label>
                  <span>{user.height} cm</span>
                </div>
                <div className="info-item">
                  <label>Weight:</label>
                  <span>{user.weight} kg</span>
                </div>
                <div className="info-item">
                  <label>BMI:</label>
                  <span>
                    {calculateBMI()}
                    {getBMICategory(calculateBMI()) && (
                      <small> ({getBMICategory(calculateBMI())})</small>
                    )}
                  </span>
                </div>
                <div className="info-item">
                  <label>Activity Level:</label>
                  <span style={{ textTransform: 'capitalize' }}>
                    {user.activity_level?.replace('_', ' ')}
                  </span>
                </div>
              </div>
            </div>
            
            <div className="profile-section">
              <h3>Health Goals</h3>
              {user.health_goals?.length > 0 ? (
                <div className="tags-list">
                  {user.health_goals.map((goal, index) => (
                    <span key={index} className="tag goal-tag">
                      {goal.replace('_', ' ')}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="no-data">No health goals set</p>
              )}
            </div>
            
            {user.medical_conditions?.length > 0 && (
              <div className="profile-section">
                <h3>Medical Conditions</h3>
                <div className="tags-list">
                  {user.medical_conditions.map((condition, index) => (
                    <span key={index} className="tag condition-tag">
                      {condition}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {user.allergies?.length > 0 && (
              <div className="profile-section">
                <h3>Allergies</h3>
                <div className="tags-list">
                  {user.allergies.map((allergy, index) => (
                    <span key={index} className="tag allergy-tag">
                      {allergy}
                    </span>
                  ))}
                </div>
              </div>
            )}

            <div className="profile-section">
              <h3>Account Information</h3>
              <div className="info-grid">
                <div className="info-item">
                  <label>Member Since:</label>
                  <span>{formatDate(user.created_at)}</span>
                </div>
                {user.updated_at && (
                  <div className="info-item">
                    <label>Last Updated:</label>
                    <span>{formatDate(user.updated_at)}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Profile;