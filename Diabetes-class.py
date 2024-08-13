st.title('Diabetes Classifier')

st.info("The goal of this GUI is to explore the data and display the results of the classification models using the 2015 CDC BRFSS data")
st.caption("HELP")
st.caption("When the dropdown menus are selected the corresponding graph will be displayed")

with st.expander('Data Description'):
    st.caption("Descritption of features")
    chart_data1 = pd.DataFrame(df_desc)
    chart_data1

with st.expander('Dataframe'):
    st.caption("Cleaned, Balanced Data with feature selection")
    st.caption("For security purposes all identifying patient information has been removed")
    chart_data = pd.DataFrame(Selected_df)
    chart_data

with st.expander('Unbalanced Class Data'):
    st.caption("Distribution of classes before over sampling")
    visual = st.pyplot(fig)

with st.expander('Balanced Class Data'):
    st.caption("Distribution of classes after over sampling")
    visual1 = st.pyplot(fig1)

with st.expander('Correlation Matrix'):
    st.caption("Correlation matrix used to determine feature selection")
    visual2 = st.pyplot(fig2)

with st.expander('Density Barcharts'):
    st.caption("Density charts for highly correlated data points")
    visual3 = st.pyplot(fig3)

st.sidebar.subheader("Choose classifier")
st.sidebar.caption("HELP")
st.sidebar.caption("Click the corresponding button to run the classifier and display the precision metrics.")

if st.sidebar.button("Classify KNN", key="classify KNN"):
        classifier_one = KNeighborsClassifier(n_neighbors=5)
        classifier_one.fit(X_train, y_train)
        accuracy = classifier_one.score(X_test, y_test)
        y_pred_two = classifier_one.predict(X_test)

        st.write("Classification Report KNN: ",classification_report(y_test, y_pred_one))
        st.write("Accuracy KNN: ", accuracy_score(y_test, y_pred_one))
        st.write("Precision KNN: ", precision_score(y_test, y_pred_one, average='micro'))
        st.write("Recall KNN: ", recall_score(y_test, y_pred_one,  average='weighted'))
        visual4 = st.pyplot(fig5)

if st.sidebar.button("Classify RF", key="classify RF"):
        classifier_two = RandomForestClassifier(n_estimators=10)
        classifier_two.fit(X_train, y_train)
        accuracy = classifier_two.score(X_test, y_test)
        y_pred_two = classifier_two.predict(X_test)

        st.write("Classification Report Random Forest: ",classification_report(y_test, y_pred_two))
        st.write("Accuracy Random Forest: ", accuracy_score(y_test, y_pred_two))
        st.write("Precision Random Forest: ", precision_score(y_test, y_pred_two, average='micro'))
        st.write("Recall Random Forest: ", recall_score(y_test, y_pred_two,  average='weighted'))
        visual5 = st.pyplot(fig6)

if st.sidebar.button("Classify MLP Classifier", key="classify MLP Classifier"):
        classifier_three = MLPClassifier(activation='logistic', solver='adam', alpha=0.0001, max_iter=1000,
                                     hidden_layer_sizes=(10,))
        classifier_three.fit(X_train, y_train)
        accuracy = classifier_three.score(X_test, y_test)
        y_pred_two = classifier_three.predict(X_test)

        st.write("Classification Report MLP Classifier: ",classification_report(y_test, classifier_three))
        st.write("Accuracy MLP Classifier: ", accuracy_score(y_test, classifier_three))
        st.write("Precision MLP Classifier: ", precision_score(y_test, classifier_three, average='micro'))
        st.write("Recall MLP Classifier: ", recall_score(y_test, classifier_three,  average='weighted'))
        visual6 = st.pyplot(fig7)
