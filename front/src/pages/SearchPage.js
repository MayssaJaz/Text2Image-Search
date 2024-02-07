import React, { useState } from 'react';
import { Input, Card, Layout } from 'antd';
import './SearchPage.css';
import FetchImages from '../apis/FetchImages';

const { Search } = Input;
const { Header, Content, Footer } = Layout;

const ImageCard = ({ imageUrl, title }) => (
    <Card
        hoverable
        className="image-card" // Apply custom CSS class for styling
        cover={<img alt={title} src={imageUrl} className="image" />} // Apply custom CSS class for styling
    >
        <Card.Meta title={title} />
    </Card>
);

const SearchPage = () => {
    const [searchTerm, setSearchTerm] = useState('');
    const [loading, setLoading] = useState(false);
    const [cards, setCards] = useState([]);

    const handleSearch = async (value) => {
        const selected_cards = [];
        const imagesUrls = await FetchImages(value, setLoading);
        for (let i = 0; i < imagesUrls.length; i++) {
            const dict = { imageUrl: imagesUrls[i], title: 'Result' + (i + 1).toString() };
            selected_cards.push(dict);
        }
        setCards(selected_cards);
    };

    return (
        <Layout className="layout">
            <Header className="header">
                <h1 style={{ color: 'white' }}>Image Search</h1>
            </Header>
            <Content className="content">
                <div className="search-container">
                    <Search
                        placeholder="Search"
                        enterButton
                        value={searchTerm}
                        onChange={e => setSearchTerm(e.target.value)}
                        onSearch={handleSearch} // Trigger search on pressing Enter
                        className="search-input" // Apply custom CSS class for styling
                    />
                </div>
                <div className="card-container"> {/* Apply custom CSS class for styling */}
                    {loading && <p>Loading...</p>}
                    {!loading && cards.map((card, index) => (
                        <ImageCard key={index} imageUrl={card.imageUrl} title={card.title} />
                    ))}
                </div>
            </Content>
            <Footer className="footer">Made with ❤️ by Mayssa</Footer>
        </Layout>
    );
};

export default SearchPage;